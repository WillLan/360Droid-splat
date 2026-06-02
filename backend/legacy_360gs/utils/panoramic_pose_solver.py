"""
Spherical 3D-2D RANSAC pose solver for ERP panoramic cameras.

Solves relative pose T_{ref鈫抍ur} given:
  - A metric radial depth map on the reference frame  (from DAP)
  - 2-D鈫?-D pixel correspondences across two ERP frames (from SphereGlue)

The solver lifts reference pixels to 3-D using ERP bearing vectors and radial
depth, then minimises the angular reprojection error on the unit sphere via
RANSAC + Levenberg-Marquardt refinement.

ERP / camera convention is kept consistent with the panoramic rendering path:
  - +Z forward
  - +X right
  - +Y down

All operations are in PyTorch (GPU-compatible).
"""

import math
import torch
import torch.nn.functional as F

from backend.legacy_360gs.utils.erp_geometry import erp_uv_to_bearing_torch as erp_uv_to_bearing


# ============================================================
# Basic geometry utilities
# ============================================================

def skew(v: torch.Tensor) -> torch.Tensor:
    """
    v: (..., 3)
    return: (..., 3, 3)
    """
    o = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    K = torch.stack([
        torch.stack([o, -vz, vy], dim=-1),
        torch.stack([vz, o, -vx], dim=-1),
        torch.stack([-vy, vx, o], dim=-1),
    ], dim=-2)
    return K


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """
    omega: (3,)
    return R: (3,3)
    """
    dtype = omega.dtype
    device = omega.device

    theta = torch.linalg.norm(omega)
    I = torch.eye(3, dtype=dtype, device=device)

    if theta < 1e-10:
        return I + skew(omega)

    K = skew(omega / theta)
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R


def se3_exp(xi: torch.Tensor) -> torch.Tensor:
    """
    xi: (6,) = [omega(3), upsilon(3)]
    return T: (4,4)
    left-multiplicative increment
    """
    dtype = xi.dtype
    device = xi.device

    omega = xi[:3]
    upsilon = xi[3:]

    theta = torch.linalg.norm(omega)
    I = torch.eye(3, dtype=dtype, device=device)

    if theta < 1e-10:
        R = I + skew(omega)
        V = I + 0.5 * skew(omega)
    else:
        K = skew(omega / theta)
        R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)

        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / (theta * theta)
        C = (theta - torch.sin(theta)) / (theta ** 3)

        Omega = skew(omega)
        V = I + B * Omega + C * (Omega @ Omega)

    t = V @ upsilon

    T = torch.eye(4, dtype=dtype, device=device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def apply_transform(T: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    T: (4,4)
    X: (N,3)
    return Y: (N,3)
    """
    R = T[:3, :3]
    t = T[:3, 3]
    return X @ R.T + t.unsqueeze(0)


# ============================================================
# ERP geometry  (bearing: utils.erp_geometry.erp_uv_to_bearing_torch)
# ============================================================

def build_tangent_basis(b: torch.Tensor) -> torch.Tensor:
    """
    Build an orthonormal tangent plane basis at each bearing vector.

    b: (N,3), unit bearings
    return B: (N,3,2)  where  B_i = [u_i, v_i]  and  B_i^T b_i = 0
    """
    N = b.shape[0]
    dtype = b.dtype
    device = b.device

    a1 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)
    a2 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).unsqueeze(0).repeat(N, 1)

    use_a1 = (torch.abs(b[:, 1]) < 0.9).unsqueeze(-1)
    a = torch.where(use_a1, a1, a2)

    u = a - (a * b).sum(dim=-1, keepdim=True) * b
    u = u / torch.clamp(torch.linalg.norm(u, dim=-1, keepdim=True), min=1e-12)

    v = torch.cross(b, u, dim=-1)
    v = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=1e-12)

    B = torch.stack([u, v], dim=-1)   # (N,3,2)
    return B


# ============================================================
# Correspondence construction
# ============================================================

def _sample_depth_and_valid_bilinear(
    depth_ref: torch.Tensor,
    valid_mask_ref: torch.Tensor,
    uv_ref: torch.Tensor,
):
    """Bilinearly sample depth at subpixel match locations.

    SphereGlue matches are subpixel.  Rounding them to integer pixels injects a
    systematic depth noise into PnP, especially around building edges.  We keep
    a correspondence only when all bilinear support pixels are valid.
    """
    H, W = depth_ref.shape
    if uv_ref.numel() == 0:
        return depth_ref.new_zeros((0,)), torch.zeros((0,), dtype=torch.bool, device=depth_ref.device)

    x = uv_ref[:, 0].clamp(0, W - 1)
    y = uv_ref[:, 1].clamp(0, H - 1)
    grid = torch.stack(
        [
            2.0 * x / max(W - 1, 1) - 1.0,
            2.0 * y / max(H - 1, 1) - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 2)
    depth_img = depth_ref.view(1, 1, H, W)
    valid_img = valid_mask_ref.to(dtype=depth_ref.dtype).view(1, 1, H, W)
    sampled_depth = F.grid_sample(
        depth_img,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).view(-1)
    sampled_valid = F.grid_sample(
        valid_img,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(-1)
    return sampled_depth, sampled_valid > 0.999

def build_spherical_3d2d_correspondences(
    depth_ref: torch.Tensor,
    valid_mask_ref: torch.Tensor,
    matches_uv: torch.Tensor,
    depth_min: float = 0.1,
    depth_max: float = 80.0,
):
    """
    Lift reference-frame pixel matches to 3-D using radial depth.

    Args:
        depth_ref      : (H, W) float,  metric radial depth [m]
        valid_mask_ref : (H, W) bool
        matches_uv     : (N, 4) float,  [u_ref, v_ref, u_cur, v_cur]
        depth_min/max  : depth filter range

    Returns:
        X_ref      : (M, 3)  鈥?3-D points in reference camera frame (Y-down)
        b_cur      : (M, 3)  鈥?unit bearing vectors in current frame
        keep_idx   : (M,)    鈥?indices into the original N matches
        (all None if no valid correspondences)
    """
    assert depth_ref.ndim == 2
    H, W = depth_ref.shape
    device = depth_ref.device
    dtype = depth_ref.dtype

    matches_uv = matches_uv.to(device=device, dtype=dtype)

    uv_ref = matches_uv[:, :2]
    uv_cur = matches_uv[:, 2:4]

    d_ref, m_ref = _sample_depth_and_valid_bilinear(depth_ref, valid_mask_ref, uv_ref)

    keep = m_ref & torch.isfinite(d_ref) & (d_ref > depth_min) & (d_ref < depth_max)
    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)

    if keep_idx.numel() == 0:
        return None, None, None

    uv_ref_keep = uv_ref[keep_idx]
    uv_cur_keep = uv_cur[keep_idx]
    d_ref_keep  = d_ref[keep_idx]

    b_ref = erp_uv_to_bearing(uv_ref_keep, W, H)  # (M,3)
    b_cur = erp_uv_to_bearing(uv_cur_keep, W, H)  # (M,3)

    X_ref = d_ref_keep.unsqueeze(-1) * b_ref       # radial depth 脳 bearing (Y-down)

    return X_ref, b_cur, keep_idx


# ============================================================
# Residuals and Jacobians
# ============================================================

def angular_errors(T: torch.Tensor, X_ref: torch.Tensor, b_cur: torch.Tensor) -> torch.Tensor:
    """
    Angular reprojection error on the unit sphere.

    T    : (4,4)   w2c transform
    X_ref: (N,3)   3-D points in ref frame
    b_cur: (N,3)   unit bearings in cur frame
    return theta: (N,) radians
    """
    Y = apply_transform(T, X_ref)
    Y_hat = Y / torch.clamp(torch.linalg.norm(Y, dim=-1, keepdim=True), min=1e-12)
    dots = (Y_hat * b_cur).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(dots)


def compute_residual_and_jacobian(T: torch.Tensor, X_ref: torch.Tensor, b_cur: torch.Tensor):
    """
    Tangent-space residual r_i = B_i^T 欧_i  and  Jacobian J_i 鈭?R^{2脳6}.

    Returns:
        r : (N, 2)
        J : (N, 2, 6)
    """
    dtype = X_ref.dtype
    device = X_ref.device

    Y   = apply_transform(T, X_ref)             # (N,3)
    Yn  = torch.clamp(torch.linalg.norm(Y, dim=-1, keepdim=True), min=1e-12)
    Y_hat = Y / Yn

    B  = build_tangent_basis(b_cur)             # (N,3,2)
    r  = torch.einsum('nij,ni->nj', B, Y_hat)  # (N,2)

    N = X_ref.shape[0]
    I3 = torch.eye(3, dtype=dtype, device=device).unsqueeze(0).repeat(N, 1, 1)
    proj   = I3 - Y_hat.unsqueeze(-1) @ Y_hat.unsqueeze(-2)   # (N,3,3)
    J_norm = proj / Yn.unsqueeze(-1)                           # (N,3,3)

    Y_skew = skew(Y)                                           # (N,3,3)
    J_pose = torch.cat([-Y_skew, I3], dim=-1)                 # (N,3,6)

    BT = B.transpose(1, 2)                                     # (N,2,3)
    J  = BT @ J_norm @ J_pose                                  # (N,2,6)

    return r, J


# ============================================================
# Robust Levenberg-Marquardt refinement
# ============================================================

def huber_weights_from_residual_norm(res_norm: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    w = torch.ones_like(res_norm)
    large = res_norm > delta
    w[large] = delta / torch.clamp(res_norm[large], min=1e-12)
    return w


def refine_pose_lm(
    T_init: torch.Tensor,
    X_ref: torch.Tensor,
    b_cur: torch.Tensor,
    num_iters: int = 15,
    lm_lambda: float = 1e-4,
    huber_delta: float = 0.05,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Levenberg-Marquardt pose refinement minimising sum ||B_i^T 欧_i||虏.
    Vectorised normal-equation assembly (no Python loop over points).

    T_init: (4,4)
    X_ref : (N,3)
    b_cur : (N,3)
    return T: (4,4)
    """
    T = T_init.clone()
    if weights is not None:
        weights = weights.to(device=X_ref.device, dtype=X_ref.dtype).view(-1)
        if weights.shape[0] != X_ref.shape[0]:
            weights = None

    for _ in range(num_iters):
        r, J = compute_residual_and_jacobian(T, X_ref, b_cur)  # r:(N,2), J:(N,2,6)

        r_norm = torch.linalg.norm(r, dim=-1)                  # (N,)
        w = huber_weights_from_residual_norm(r_norm, delta=huber_delta)  # (N,)
        if weights is not None:
            w = w * weights.clamp_min(1e-4)

        # Vectorised: H = sum_i w_i J_i^T J_i  鈫? einsum
        wJ = J * w[:, None, None]                               # (N,2,6)
        H  = torch.einsum('nij,nik->jk', wJ, J)                # (6,6)
        g  = torch.einsum('nij,ni->j',   wJ, r)                # (6,)

        H = H + lm_lambda * torch.eye(6, dtype=T.dtype, device=T.device)

        try:
            delta_xi = -torch.linalg.solve(H, g)
        except RuntimeError:
            break

        if torch.linalg.norm(delta_xi) < 1e-8:
            break

        T = se3_exp(delta_xi) @ T

    return T


# ============================================================
# RANSAC main solver
# ============================================================

def solve_pose_spherical_3d2d_points_ransac(
    X_ref: torch.Tensor,
    b_cur: torch.Tensor,
    sample_size: int = 6,
    max_ransac_iters: int = 1000,
    tau_ang_deg: float = 2.0,
    lm_iters_hypothesis: int = 10,
    lm_iters_final: int = 20,
    dtype: torch.dtype = torch.float64,
    T_init: torch.Tensor | None = None,
    correspondence_weights: torch.Tensor | None = None,
    debug_record_samples: bool = False,
):
    """RANSAC + LM pose solve from 3-D points to current-frame bearings.

    ``X_ref`` can be in any source coordinate frame as long as the returned
    transform maps that frame into the current camera.  The legacy two-view
    solver passes reference-camera points and receives ``T_ref_to_cur``; the
    multi-reference frontend passes world points and receives absolute w2c.
    """
    if X_ref is None or b_cur is None:
        return None, None, None, None, {"samples": []}
    if X_ref.shape[0] < sample_size or b_cur.shape[0] != X_ref.shape[0]:
        return None, None, X_ref, b_cur, {"samples": []}

    device = X_ref.device
    X_ref = X_ref.to(device=device, dtype=dtype)
    b_cur = b_cur.to(device=device, dtype=dtype)
    M = X_ref.shape[0]
    tau_ang = math.radians(tau_ang_deg)
    corr_w = None
    if correspondence_weights is not None:
        corr_w = correspondence_weights.to(device=device, dtype=dtype).view(-1)
        if corr_w.numel() == M:
            corr_w = corr_w.clamp_min(1e-4)
            corr_w = corr_w / corr_w.mean().clamp_min(1e-6)
        else:
            corr_w = None

    T_identity = torch.eye(4, dtype=dtype, device=device)
    T_prior = T_init.to(device=device, dtype=dtype) if T_init is not None else T_identity
    best_T = None
    best_inlier_mask = None
    best_num_inliers = -1.0
    best_score = float("inf")

    debug_samples = []

    with torch.no_grad():
        def _candidate_score(theta: torch.Tensor, inlier_mask: torch.Tensor):
            num_inliers = int(inlier_mask.sum().item())
            if num_inliers < sample_size:
                return 0.0, float("inf")
            if corr_w is None:
                return float(num_inliers), theta[inlier_mask].mean().item()
            weighted_inliers = float(corr_w[inlier_mask].sum().item())
            weighted_error = (
                theta[inlier_mask] * corr_w[inlier_mask]
            ).sum() / corr_w[inlier_mask].sum().clamp_min(1e-6)
            return weighted_inliers, weighted_error.item()

        for T_seed in (T_prior, T_identity):
            T_cand = refine_pose_lm(
                T_init=T_seed,
                X_ref=X_ref,
                b_cur=b_cur,
                num_iters=lm_iters_hypothesis,
                lm_lambda=1e-4,
                huber_delta=0.05,
                weights=corr_w,
            )
            theta = angular_errors(T_cand, X_ref, b_cur)
            inlier_mask = theta < tau_ang
            num_inliers, score = _candidate_score(theta, inlier_mask)
            if (num_inliers > best_num_inliers) or (
                num_inliers == best_num_inliers and score < best_score
            ):
                best_num_inliers = num_inliers
                best_score = score
                best_T = T_cand
                best_inlier_mask = inlier_mask.clone()

        for iter_idx in range(max_ransac_iters):
            if corr_w is not None:
                sample_idx = torch.multinomial(
                    corr_w.clamp_min(1e-4), sample_size, replacement=False
                )
            else:
                sample_idx = torch.randperm(M, device=device)[:sample_size]
            if debug_record_samples:
                debug_samples.append(
                    {
                        "iter": int(iter_idx),
                        "valid_indices": sample_idx.detach().cpu().tolist(),
                    }
                )
            X_s = X_ref[sample_idx]
            b_s = b_cur[sample_idx]
            w_s = corr_w[sample_idx] if corr_w is not None else None

            T_cand = refine_pose_lm(
                T_init=T_prior,
                X_ref=X_s,
                b_cur=b_s,
                num_iters=lm_iters_hypothesis,
                lm_lambda=1e-4,
                huber_delta=0.05,
                weights=w_s,
            )

            theta = angular_errors(T_cand, X_ref, b_cur)
            inlier_mask = theta < tau_ang
            num_inliers, score = _candidate_score(theta, inlier_mask)
            if num_inliers < sample_size:
                continue

            if (num_inliers > best_num_inliers) or (
                num_inliers == best_num_inliers and score < best_score
            ):
                best_num_inliers = num_inliers
                best_score = score
                best_T = T_cand
                best_inlier_mask = inlier_mask.clone()
                if debug_record_samples and debug_samples:
                    debug_samples[-1]["became_best"] = True
                    debug_samples[-1]["num_inliers"] = float(num_inliers)
                    debug_samples[-1]["score"] = float(score)

        if best_T is None or best_inlier_mask is None or best_inlier_mask.sum() < sample_size:
            return None, None, X_ref, b_cur, {"samples": debug_samples}

        X_in = X_ref[best_inlier_mask]
        b_in = b_cur[best_inlier_mask]
        T_best = refine_pose_lm(
            T_init=best_T,
            X_ref=X_in,
            b_cur=b_in,
            num_iters=lm_iters_final,
            lm_lambda=1e-4,
            huber_delta=0.05,
            weights=corr_w[best_inlier_mask] if corr_w is not None else None,
        )

        theta_final = angular_errors(T_best, X_ref, b_cur)
        inlier_mask_final = theta_final < tau_ang

    return T_best, inlier_mask_final, X_ref, b_cur, {"samples": debug_samples}


def solve_pose_spherical_3d2d_ransac(
    depth_ref: torch.Tensor,
    valid_mask_ref: torch.Tensor,
    matches_uv: torch.Tensor,
    depth_min: float = 0.1,
    depth_max: float = 80.0,
    sample_size: int = 6,
    max_ransac_iters: int = 1000,
    tau_ang_deg: float = 2.0,
    lm_iters_hypothesis: int = 10,
    lm_iters_final: int = 20,
    dtype: torch.dtype = torch.float64,
    T_init: torch.Tensor | None = None,
    correspondence_weights: torch.Tensor | None = None,
    debug_record_samples: bool = False,
):
    """
    Full RANSAC + LM spherical 3D-2D pose solver.

    Args:
        depth_ref      : (H, W) radial depth [m]
        valid_mask_ref : (H, W) bool
        matches_uv     : (N, 4) [u_ref, v_ref, u_cur, v_cur]

    Returns (all None on failure):
        T_best          : (4,4) estimated T_{ref鈫抍ur}
        inlier_mask     : (M,) bool  final inlier mask over valid correspondences
        X_ref_valid     : (M,3) 3-D ref points
        b_cur_valid     : (M,3) cur bearings
        keep_idx        : indices into original matches
    """
    device = depth_ref.device
    depth_ref      = depth_ref.to(dtype=dtype)
    valid_mask_ref = valid_mask_ref.to(device=device)

    X_ref, b_cur, keep_idx = build_spherical_3d2d_correspondences(
        depth_ref=depth_ref,
        valid_mask_ref=valid_mask_ref,
        matches_uv=matches_uv,
        depth_min=depth_min,
        depth_max=depth_max,
    )

    if X_ref is None or X_ref.shape[0] < sample_size:
        return None, None, None, None, None, {"samples": []}

    corr_w = None
    if correspondence_weights is not None and keep_idx is not None:
        corr_w = correspondence_weights.to(device=device, dtype=dtype).view(-1)
        if corr_w.numel() >= int(keep_idx.max().item()) + 1:
            corr_w = corr_w[keep_idx].clamp_min(1e-4)
            corr_w = corr_w / corr_w.mean().clamp_min(1e-6)
        else:
            corr_w = None

    T_best, inlier_mask_final, X_ref, b_cur, debug_info = solve_pose_spherical_3d2d_points_ransac(
        X_ref=X_ref,
        b_cur=b_cur,
        sample_size=sample_size,
        max_ransac_iters=max_ransac_iters,
        tau_ang_deg=tau_ang_deg,
        lm_iters_hypothesis=lm_iters_hypothesis,
        lm_iters_final=lm_iters_final,
        dtype=dtype,
        T_init=T_init,
        correspondence_weights=corr_w,
        debug_record_samples=debug_record_samples,
    )
    if debug_record_samples and debug_info is not None:
        for sample in debug_info.get("samples", []):
            valid_indices = sample.get("valid_indices", None)
            if valid_indices is not None:
                idx = torch.as_tensor(valid_indices, dtype=torch.long, device=keep_idx.device)
                sample["match_indices"] = keep_idx[idx].detach().cpu().tolist()

    return T_best, inlier_mask_final, X_ref, b_cur, keep_idx, debug_info
