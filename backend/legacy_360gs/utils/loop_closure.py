"""
Panoramic loop closure detection and pose-graph optimisation for GS-SLAM.

Pipeline (2-step):
  1. Appearance-based candidate retrieval:
       ``PanoLoopDetector.query(frame_idx, image)``
       Encodes the ERP image as a compact descriptor (default: flatten a
       down-scaled grayscale thumbnail + L2-normalise 鈫?cosine similarity).
       When torchvision is available and ``use_dino_features=True`` in config,
       a lightweight DINO/ViT feature is used instead for better invariance.

  2. Geometric verification:
       ``PanoLoopDetector.verify(query_idx, cand_idx, ...)``
       Uses the already-implemented ``solve_pose_spherical_3d2d_ransac`` with
       pre-cached SphereGlue matches to confirm the loop and estimate the
       relative pose.

  3. Pose-graph optimisation:
       ``pose_graph_optimize(nodes, edges)``
       Minimises 危 ||log(T_ij^{-1} 路 T_i^{-1} 路 T_j)||虏 over all SE(3) node
       poses using scipy ``least_squares``.  The implementation uses a simple
       Lie-algebra parameterisation (se3 vector, left perturbation) with a
       Huber robust kernel.

  4. Gaussian map correction:
       ``correct_gaussian_map(gaussians, old_poses, new_poses, kf_viewpoints)``
       For each keyframe whose pose changed by more than a threshold, the
       world-space positions of Gaussians that were *primarily* observed from
       that keyframe (based on ``unique_kfIDs``) are updated via a rigid
       SE(3) transform delta.

All functionality is gated by ``enable_loop_closure: False`` in the config.
When False, ``PanoLoopDetector.query`` is a no-op and returns an empty list.

Usage (from slam_frontend.py):
    detector = PanoLoopDetector(config)
    ...  # per frame
    candidates = detector.query(frame_idx, erp_image)
    for cand_idx in candidates:
        rel_pose, ok = detector.verify(frame_idx, cand_idx, depth_map, matches)
        if ok:
            detector.add_loop_edge(frame_idx, cand_idx, rel_pose)
    ...  # after N keyframes
    if detector.should_optimize():
        new_poses = pose_graph_optimize(detector.get_graph())
        correct_gaussian_map(gaussians, detector.get_old_poses(), new_poses, viewpoints)
        detector.update_poses(new_poses)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small SE(3) helpers (numpy-only, no torch dependency for the optimiser)
# ---------------------------------------------------------------------------

def _so3_exp(w: np.ndarray) -> np.ndarray:
    """Rodriguez formula: SO(3) exp map, w in R^3."""
    angle = np.linalg.norm(w)
    if angle < 1e-8:
        return np.eye(3) + _skew(w)
    axis = w / angle
    K = _skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) log map 鈫?R^3."""
    cos_a = (np.trace(R) - 1.0) / 2.0
    cos_a = np.clip(cos_a, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if abs(angle) < 1e-8:
        return np.zeros(3)
    return angle / (2.0 * np.sin(angle)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    """SE(3) exp map, xi = [rho, theta] in R^6 鈫?4脳4 matrix."""
    rho   = xi[:3]
    theta = xi[3:]
    R = _so3_exp(theta)
    angle = np.linalg.norm(theta)
    if angle < 1e-8:
        V = np.eye(3)
    else:
        K = _skew(theta / angle)
        V = (np.eye(3)
             + (1 - np.cos(angle)) / angle * K
             + (angle - np.sin(angle)) / angle * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = V @ rho
    return T


def _se3_log(T: np.ndarray) -> np.ndarray:
    """SE(3) log map, 4脳4 matrix 鈫?R^6 [rho, theta]."""
    R = T[:3, :3]
    t = T[:3,  3]
    theta_vec = _so3_log(R)
    angle = np.linalg.norm(theta_vec)
    if angle < 1e-8:
        V_inv = np.eye(3)
    else:
        K = _skew(theta_vec / angle)
        V_inv = (np.eye(3)
                 - 0.5 * K
                 + (1.0 / angle**2) * (1 - angle * np.sin(angle) / (2 * (1 - np.cos(angle)))) * (K @ K))
    rho = V_inv @ t
    return np.concatenate([rho, theta_vec])


def _T_from_R_t(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ---------------------------------------------------------------------------
# Appearance descriptor
# ---------------------------------------------------------------------------

def _thumbnail_descriptor(image: torch.Tensor, size: Tuple[int, int] = (16, 32)) -> np.ndarray:
    """Compute a fast L2-normalised grayscale histogram descriptor.

    Args:
        image: (3, H, W) float32 ERP image in [0, 1].
        size:  (H_thumb, W_thumb) thumbnail size.

    Returns:
        1-D float64 array, L2-normalised.
    """
    import torch.nn.functional as F
    H_t, W_t = size
    thumb = F.interpolate(
        image.unsqueeze(0), size=(H_t, W_t), mode="bilinear", align_corners=False
    ).squeeze(0)                                              # (3, H_t, W_t)
    gray = thumb.mean(dim=0).cpu().numpy().astype(np.float64)  # (H_t, W_t)
    desc = gray.reshape(-1)
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc /= norm
    return desc


def _spherical_band_descriptor(
    image: torch.Tensor,
    bands: int = 8,
    width_bins: int = 16,
) -> np.ndarray:
    """Lightweight panoramic descriptor with latitude-band pooling.

    This is used as the practical backend for ``dinov2_spherical`` when a
    real DINOv2 model is not cached locally. It preserves ERP band structure
    better than a flat thumbnail descriptor and is deterministic/offline.
    """
    import torch.nn.functional as F

    pooled = F.interpolate(
        image.unsqueeze(0), size=(bands, width_bins), mode="bilinear", align_corners=False
    ).squeeze(0)
    band_mean = pooled.mean(dim=0).reshape(-1).cpu().numpy().astype(np.float64)
    band_std = pooled.std(dim=0).reshape(-1).cpu().numpy().astype(np.float64)
    desc = np.concatenate([band_mean, band_std], axis=0)
    norm = np.linalg.norm(desc)
    if norm > 1e-8:
        desc /= norm
    return desc


_DINO_LOCAL_REPO = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov2_main")
_DINO_CHECKPOINTS = {
    "dinov2_vits14": os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth"),
    "dinov2_vitb14": os.path.expanduser("~/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"),
}


def _load_local_dinov2_backbone(
    model_name: str = "dinov2_vits14",
    device: str = "cpu",
):
    ckpt = _DINO_CHECKPOINTS.get(model_name, "")
    if not os.path.isdir(_DINO_LOCAL_REPO) or not ckpt or not os.path.isfile(ckpt):
        return None
    try:
        model = torch.hub.load(
            _DINO_LOCAL_REPO,
            model_name,
            source="local",
            pretrained=True,
        )
        model = model.to(device)
        model.eval()
        return model
    except Exception as exc:
        logger.warning(f"[LoopClosure] DINOv2 load failed for {model_name}: {exc}")
        return None


def _dinov2_spherical_descriptor(
    image: torch.Tensor,
    model,
    device: str = "cpu",
    input_hw: Tuple[int, int] = (224, 448),
    bands: int = 8,
) -> np.ndarray:
    if model is None:
        raise ValueError("DINOv2 model is required for dinov2_spherical descriptor.")
    img = image.unsqueeze(0).to(device=device, dtype=torch.float32)
    img = F.interpolate(img, size=input_hw, mode="bilinear", align_corners=False)
    with torch.no_grad():
        features = model.forward_features(img)
    patch_tokens = features["x_norm_patchtokens"]  # (1, N, C)
    h_tokens = max(1, input_hw[0] // 14)
    w_tokens = max(1, input_hw[1] // 14)
    patch_tokens = patch_tokens.reshape(1, h_tokens, w_tokens, -1)[0]
    band_chunks = torch.chunk(patch_tokens, max(1, int(bands)), dim=0)
    pooled = []
    for chunk in band_chunks:
        if chunk.numel() == 0:
            continue
        n_rows = chunk.shape[0]
        lat = torch.linspace(-0.5, 0.5, steps=n_rows, device=chunk.device, dtype=chunk.dtype)
        weights = torch.cos(lat * np.pi).clamp_min(1e-3).view(n_rows, 1, 1)
        band_feat = (chunk * weights).sum(dim=(0, 1)) / weights.sum().clamp_min(1e-6)
        pooled.append(band_feat)
    if not pooled:
        raise RuntimeError("Empty pooled DINOv2 descriptor.")
    desc = torch.cat(pooled, dim=0)
    desc = F.normalize(desc, dim=0)
    return desc.detach().cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Loop detector
# ---------------------------------------------------------------------------

class PanoLoopDetector:
    """Appearance-based panoramic loop detector with geometric verification.

    Config keys (under ``Training``):
        enable_loop_closure         bool  (default False)
        loop_desc_size              tuple (default [16, 32])
        loop_cos_threshold          float (default 0.92)
        loop_min_frame_gap          int   (default 30)   鈥?minimum frames between query and candidate
        loop_ransac_inlier_ratio    float (default 0.35)
        loop_pg_optimize_every      int   (default 20)   鈥?run PGO every N new loop edges
        loop_pg_max_iters           int   (default 200)
        loop_map_correct_threshold  float (default 0.05) 鈥?metres; poses with smaller delta skipped
    """

    def __init__(self, config: dict):
        self.config = config
        self._enabled = bool(
            config.get("Training", {}).get("enable_loop_closure", False)
        )
        training_cfg = config.get("Training", {})
        desc_size_cfg = training_cfg.get("loop_desc_size", [16, 32])
        self._desc_size = tuple(desc_size_cfg)
        self._cos_thresh = float(training_cfg.get("loop_cos_threshold", 0.92))
        self._min_gap    = int(training_cfg.get("loop_min_frame_gap", 30))
        self._ransac_inlier = float(training_cfg.get("loop_ransac_inlier_ratio", 0.35))
        self._pg_every   = int(training_cfg.get("loop_pg_optimize_every", 20))
        self._descriptor_type = str(training_cfg.get("loop_descriptor", "thumbnail")).lower()
        self._dino_model_name = str(training_cfg.get("loop_dino_model_name", "dinov2_vits14"))
        self._dino_device = str(training_cfg.get("loop_dino_device", "cpu"))
        self._dinov2_model = None
        self._effective_descriptor_type = self._descriptor_type

        # Database: frame_idx 鈫?descriptor (np.ndarray)
        self._db: Dict[int, np.ndarray] = {}
        # Odometry poses: frame_idx 鈫?4脳4 c2w (numpy)
        self._poses_c2w: Dict[int, np.ndarray] = {}
        # Loop edges: list of (i, j, T_ij)  where T_ij = T_j^{-1} @ T_i (c2w convention)
        self._loop_edges: List[Tuple[int, int, np.ndarray]] = []
        self._n_edges_at_last_pgo = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register_keyframe(
        self,
        frame_idx: int,
        image: torch.Tensor,
        pose_c2w: np.ndarray,
    ) -> None:
        """Register a new keyframe in the descriptor database.

        Args:
            frame_idx: Global keyframe index.
            image:     (3, H, W) float32 ERP image.
            pose_c2w:  4脳4 numpy array (camera-to-world).
        """
        if not self._enabled:
            return
        self._db[frame_idx] = self._compute_descriptor(image)
        self._poses_c2w[frame_idx] = pose_c2w.copy()

    def _compute_descriptor(self, image: torch.Tensor) -> np.ndarray:
        if self._descriptor_type == "dinov2_spherical":
            if self._dinov2_model is None:
                self._dinov2_model = _load_local_dinov2_backbone(
                    self._dino_model_name, device=self._dino_device
                )
                self._effective_descriptor_type = (
                    "dinov2_spherical" if self._dinov2_model is not None else "band_pool"
                )
                logger.info(
                    "[LoopClosure] descriptor backend=%s",
                    self._effective_descriptor_type,
                )
            if self._dinov2_model is not None:
                return _dinov2_spherical_descriptor(
                    image,
                    self._dinov2_model,
                    device=self._dino_device,
                    input_hw=(224, 448),
                    bands=max(4, int(self._desc_size[0] // 2)),
                )
            return _spherical_band_descriptor(
                image,
                bands=max(4, int(self._desc_size[0] // 2)),
                width_bins=max(8, int(self._desc_size[1] // 2)),
            )
        if self._descriptor_type == "band_pool":
            return _spherical_band_descriptor(
                image,
                bands=max(4, int(self._desc_size[0] // 2)),
                width_bins=max(8, int(self._desc_size[1] // 2)),
            )
        self._effective_descriptor_type = "thumbnail"
        return _thumbnail_descriptor(image, self._desc_size)

    def query(self, frame_idx: int) -> List[int]:
        """Return candidate loop-closure keyframe indices for ``frame_idx``.

        Args:
            frame_idx: Index of the current query keyframe.

        Returns:
            Sorted list of candidate indices (ascending, may be empty).
        """
        if not self._enabled or frame_idx not in self._db:
            return []

        q_desc = self._db[frame_idx]
        candidates = []
        for idx, desc in self._db.items():
            if abs(frame_idx - idx) < self._min_gap:
                continue
            cos_sim = float(q_desc @ desc)
            if cos_sim >= self._cos_thresh:
                candidates.append((idx, cos_sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in candidates[:5]]  # top-5

    def verify(
        self,
        query_idx: int,
        cand_idx: int,
        query_depth: Optional[np.ndarray],
        matches_uv: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], bool]:
        """Geometrically verify a loop candidate using spherical RANSAC.

        Args:
            query_idx:   Query keyframe index.
            cand_idx:    Candidate keyframe index.
            query_depth: (H, W) radial depth map for the candidate frame, or None.
            matches_uv:  (N, 4) array of matched (u_ref, v_ref, u_cur, v_cur)
                         pixel coordinates, or None (skips RANSAC if None).

        Returns:
            (rel_pose, success) where rel_pose is 4脳4 T_{query鈫恈and} (c2w delta)
            and success indicates whether the inlier ratio exceeded the threshold.
        """
        if not self._enabled:
            return None, False
        if matches_uv is None or query_depth is None:
            return None, False
        if matches_uv.shape[0] < 8:
            return None, False

        try:
            from backend.legacy_360gs.utils.panoramic_pose_solver import solve_pose_spherical_3d2d_ransac
            from backend.legacy_360gs.utils.erp_geometry import erp_uv_to_bearing_numpy

            H, W = query_depth.shape
            R_est, t_est, inlier_mask = solve_pose_spherical_3d2d_ransac(
                depth_ref=query_depth,
                matches_uv=matches_uv,
                H=H, W=W,
            )
            inlier_ratio = float(inlier_mask.sum()) / max(len(inlier_mask), 1)
            if inlier_ratio < self._ransac_inlier:
                return None, False

            rel_pose = _T_from_R_t(R_est, t_est)
            return rel_pose, True
        except Exception as e:
            logger.debug(f"Loop verify failed ({query_idx}->{cand_idx}): {e}")
            return None, False

    def add_loop_edge(
        self,
        i: int,
        j: int,
        T_rel: np.ndarray,
    ) -> None:
        """Record a confirmed loop edge.

        Args:
            i:     Query frame index.
            j:     Candidate frame index.
            T_rel: 4脳4 relative pose T_{i鈫恓} (camera-space, c2w convention).
        """
        if not self._enabled:
            return
        self._loop_edges.append((i, j, T_rel))

    def should_optimize(self) -> bool:
        """Return True when enough new loop edges have accumulated."""
        new_edges = len(self._loop_edges) - self._n_edges_at_last_pgo
        return self._enabled and new_edges >= self._pg_every

    def get_graph(self):
        """Return (nodes, odom_edges, loop_edges) for pose_graph_optimize."""
        nodes = sorted(self._poses_c2w.keys())
        return nodes, self._poses_c2w, self._loop_edges

    def update_poses(self, new_poses_c2w: Dict[int, np.ndarray]) -> None:
        """Replace stored poses after PGO."""
        for idx, T in new_poses_c2w.items():
            self._poses_c2w[idx] = T.copy()
        self._n_edges_at_last_pgo = len(self._loop_edges)

    def get_old_poses(self) -> Dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self._poses_c2w.items()}

    def reset(self) -> None:
        self._db.clear()
        self._poses_c2w.clear()
        self._loop_edges.clear()
        self._n_edges_at_last_pgo = 0


# ---------------------------------------------------------------------------
# Pose-graph optimisation
# ---------------------------------------------------------------------------

def pose_graph_optimize(
    nodes: List[int],
    poses_c2w: Dict[int, np.ndarray],
    loop_edges: List[Tuple[int, int, np.ndarray]],
    config: dict,
    fix_first: bool = True,
) -> Dict[int, np.ndarray]:
    """Minimise the pose-graph cost over SE(3) node poses.

    The cost is a sum of squared se(3) residuals with a Huber kernel:

        危_{(i,j)} huber( || log( T_ij^{-1} 路 T_i^{-1} 路 T_j ) || )

    where T_ij is the measured relative pose, T_i and T_j are the optimised
    node poses (c2w), and huber delta is read from config.

    Args:
        nodes:       Ordered list of keyframe indices.
        poses_c2w:   Initial c2w poses (4脳4 numpy).
        loop_edges:  List of (i, j, T_ij) confirmed loop constraints.
        config:      SLAM config dict.
        fix_first:   Fix the first node to anchor the gauge freedom.

    Returns:
        Dict mapping frame_idx 鈫?optimised 4脳4 c2w pose.
    """
    if not loop_edges:
        return {k: v.copy() for k, v in poses_c2w.items()}

    try:
        from scipy.optimize import least_squares
    except ImportError:
        logger.warning("scipy not available; skipping pose graph optimisation.")
        return {k: v.copy() for k, v in poses_c2w.items()}

    training_cfg = config.get("Training", {})
    max_iters  = int(training_cfg.get("loop_pg_max_iters", 200))
    huber_delta = float(training_cfg.get("loop_pg_huber_delta", 0.1))

    n = len(nodes)
    idx_map = {node: i for i, node in enumerate(nodes)}

    # Initial parameter vector: concatenate se(3) vectors for each node.
    # Node 0 is fixed (6 zeros not in the variable vector when fix_first=True).
    def _pack(poses: Dict[int, np.ndarray]) -> np.ndarray:
        vecs = []
        for node in nodes:
            if fix_first and node == nodes[0]:
                continue
            T = poses[node]
            vecs.append(_se3_log(T))
        return np.concatenate(vecs) if vecs else np.zeros(0)

    def _unpack(x: np.ndarray) -> Dict[int, np.ndarray]:
        result = {}
        pos = 0
        for node in nodes:
            if fix_first and node == nodes[0]:
                result[node] = poses_c2w[nodes[0]].copy()
                continue
            xi = x[pos:pos + 6]
            result[node] = _se3_exp(xi) @ poses_c2w[nodes[0]]
            pos += 6
        return result

    def _residuals(x: np.ndarray) -> np.ndarray:
        cur_poses = _unpack(x)
        res = []
        for (i, j, T_ij_meas) in loop_edges:
            if i not in cur_poses or j not in cur_poses:
                continue
            T_i = cur_poses[i]
            T_j = cur_poses[j]
            # Predicted relative: T_{i鈫恓} = T_i^{-1} @ T_j
            T_pred = np.linalg.inv(T_i) @ T_j
            # Residual in se(3)
            delta = T_ij_meas @ np.linalg.inv(T_pred)
            xi = _se3_log(delta)
            # Huber weight
            r_norm = np.linalg.norm(xi)
            w = min(1.0, huber_delta / max(r_norm, 1e-8))
            res.append(w * xi)
        return np.concatenate(res) if res else np.zeros(1)

    x0 = _pack(poses_c2w)
    if x0.size == 0:
        return {k: v.copy() for k, v in poses_c2w.items()}

    try:
        result = least_squares(
            _residuals, x0,
            method="lm",
            max_nfev=max_iters * len(x0),
        )
        opt_poses = _unpack(result.x)
    except Exception as e:
        logger.warning(f"Pose graph optimisation failed: {e}")
        opt_poses = {k: v.copy() for k, v in poses_c2w.items()}

    return opt_poses


# ---------------------------------------------------------------------------
# Gaussian map correction
# ---------------------------------------------------------------------------

def correct_gaussian_map(
    gaussians,
    old_poses_c2w: Dict[int, np.ndarray],
    new_poses_c2w: Dict[int, np.ndarray],
    kf_viewpoints: dict,
    config: dict,
) -> None:
    """Apply rigid SE(3) corrections to Gaussian positions after PGO.

    For each keyframe whose pose changed by more than
    ``loop_map_correct_threshold``, world-space positions of Gaussians whose
    ``unique_kfIDs`` matches that keyframe are updated via:

        p_world_new = delta_T @ p_world_old

    where delta_T = T_new @ T_old^{-1}.

    Args:
        gaussians:       GaussianModel instance.
        old_poses_c2w:   Pre-PGO c2w poses (frame_idx 鈫?4脳4 numpy).
        new_poses_c2w:   Post-PGO c2w poses (frame_idx 鈫?4脳4 numpy).
        kf_viewpoints:   Dict[frame_idx 鈫?Camera/PanoramaCamera] for pose update.
        config:          SLAM config dict.
    """
    from backend.legacy_360gs.utils.pose_utils import update_pose as _update_pose
    from backend.legacy_360gs.gaussian_splatting.utils.graphics_utils import getWorld2View2

    threshold = float(
        config.get("Training", {}).get("loop_map_correct_threshold", 0.05)
    )

    xyz = gaussians.get_xyz.detach()  # (N, 3) cuda
    anchor_kf = getattr(gaussians, "_anchor_kf", None)
    anchor_submap = getattr(gaussians, "_anchor_submap", None)
    kf_ids = (
        anchor_kf
        if anchor_kf is not None and anchor_kf.shape[0] == xyz.shape[0]
        else gaussians.unique_kfIDs
    )  # (N,) CPU int

    with torch.no_grad():
        updated_mask = torch.zeros((xyz.shape[0],), device=xyz.device, dtype=torch.bool)
        submap_deltas = {}
        for frame_idx, T_new in new_poses_c2w.items():
            T_old = old_poses_c2w.get(frame_idx)
            if T_old is None:
                continue
            delta_T = T_new @ np.linalg.inv(T_old)
            delta_t = delta_T[:3, 3]
            if np.linalg.norm(delta_t) < threshold:
                continue

            # Select Gaussians observed primarily from this keyframe
            kf_mask = (kf_ids == frame_idx).to(device=xyz.device)
            if not kf_mask.any():
                continue

            # Transform world positions
            delta_R = torch.from_numpy(delta_T[:3, :3].astype(np.float32)).cuda()
            delta_t_t = torch.from_numpy(delta_T[:3, 3].astype(np.float32)).cuda()

            pts = xyz[kf_mask]  # (M, 3)
            xyz_new = (delta_R @ pts.T).T + delta_t_t
            gaussians._xyz.data[kf_mask] = xyz_new
            updated_mask = updated_mask | kf_mask
            if frame_idx in kf_viewpoints:
                submap_id = int(getattr(kf_viewpoints[frame_idx], "submap_id", -1))
                if submap_id >= 0 and submap_id not in submap_deltas:
                    submap_deltas[submap_id] = delta_T.copy()

            # Also update the camera pose stored in the viewpoint
            if frame_idx in kf_viewpoints:
                vp = kf_viewpoints[frame_idx]
                # T_new is c2w; convert to w2c (R, T)
                R_new = T_new[:3, :3]
                t_new = -R_new.T @ T_new[:3, 3]
                vp.R = torch.from_numpy(R_new.astype(np.float32))
                vp.T = torch.from_numpy(t_new.astype(np.float32))
                # Reset pose deltas
                if hasattr(vp, "cam_rot_delta"):
                    vp.cam_rot_delta.data.zero_()
                if hasattr(vp, "cam_trans_delta"):
                    vp.cam_trans_delta.data.zero_()

        if anchor_submap is not None and anchor_submap.shape[0] == xyz.shape[0]:
            for submap_id, delta_T in submap_deltas.items():
                submap_mask = (anchor_submap == submap_id).to(device=xyz.device) & (~updated_mask)
                if not submap_mask.any():
                    continue
                delta_R = torch.from_numpy(delta_T[:3, :3].astype(np.float32)).cuda()
                delta_t_t = torch.from_numpy(delta_T[:3, 3].astype(np.float32)).cuda()
                pts = xyz[submap_mask]
                xyz_new = (delta_R @ pts.T).T + delta_t_t
                gaussians._xyz.data[submap_mask] = xyz_new

    logger.info(f"[LoopClosure] Applied Gaussian map corrections for "
                f"{len(new_poses_c2w)} keyframes.")
