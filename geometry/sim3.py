"""Small, dependency-free Sim(3) utilities used by the global window graph.

Transforms are stored as 4x4 homogeneous matrices whose upper-left block is
``scale * rotation``.  Camera poses remain ordinary SE(3) matrices; callers
must use :func:`apply_sim3_to_pose` when moving a camera between frames.
"""

from __future__ import annotations

import torch

from frontend.pano_droid.spherical_ba import skew, so3_exp


def sim3_identity(*, device=None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.eye(4, device=device, dtype=dtype)


def sim3_from_components(
    scale: torch.Tensor | float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    scale_t = torch.as_tensor(scale, device=rotation.device, dtype=rotation.dtype)
    scale_t = scale_t.expand(rotation.shape[:-2])
    out = torch.zeros(*rotation.shape[:-2], 4, 4, device=rotation.device, dtype=rotation.dtype)
    out[..., :3, :3] = scale_t[..., None, None] * rotation
    out[..., :3, 3] = translation
    out[..., 3, 3] = 1.0
    return out


def project_rotation_to_so3(rotation: torch.Tensor) -> torch.Tensor:
    """Return the closest proper rotation in Frobenius norm.

    Sim(3) nodes are stored in float32 and can acquire a small shear after
    repeated graph/map synchronization.  The shear must not be interpreted as
    either camera rotation or scale because the rest of this module relies on
    ``R.T == R.inverse()``.
    """

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end in 3x3, got {tuple(rotation.shape)}"
        )
    u, _, vh = torch.linalg.svd(rotation)
    correction = torch.ones(
        *rotation.shape[:-2],
        3,
        device=rotation.device,
        dtype=rotation.dtype,
    )
    correction[..., -1] = torch.where(
        torch.linalg.det(u @ vh) < 0.0,
        rotation.new_tensor(-1.0),
        rotation.new_tensor(1.0),
    )
    return (u * correction.unsqueeze(-2)) @ vh


def sim3_components(transform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if transform.shape[-2:] != (4, 4):
        raise ValueError(f"Sim(3) transform must end in 4x4, got {tuple(transform.shape)}")
    linear = transform[..., :3, :3]
    determinant = torch.linalg.det(linear)
    scale = determinant.abs().clamp_min(torch.finfo(linear.dtype).eps).pow(1.0 / 3.0)
    rotation = linear / scale[..., None, None]
    return scale, rotation, transform[..., :3, 3]


def canonicalize_sim3(transform: torch.Tensor) -> torch.Tensor:
    """Retract a finite approximate Sim(3) matrix onto the Sim(3) manifold."""

    scale, rotation, translation = sim3_components(transform)
    return sim3_from_components(
        scale,
        project_rotation_to_so3(rotation),
        translation,
    )


def canonicalize_c2w(pose_c2w: torch.Tensor) -> torch.Tensor:
    """Retract a finite homogeneous camera pose onto SE(3)."""

    if pose_c2w.shape[-2:] != (4, 4):
        raise ValueError(
            f"pose_c2w must end in 4x4, got {tuple(pose_c2w.shape)}"
        )
    output = pose_c2w.clone()
    output[..., :3, :3] = project_rotation_to_so3(pose_c2w[..., :3, :3])
    output[..., 3, :] = 0.0
    output[..., 3, 3] = 1.0
    return output


def sim3_inverse(transform: torch.Tensor) -> torch.Tensor:
    scale, rotation, translation = sim3_components(transform)
    rotation_t = rotation.transpose(-1, -2)
    inverse_scale = scale.reciprocal()
    inverse_translation = -inverse_scale[..., None] * torch.einsum(
        "...ij,...j->...i", rotation_t, translation
    )
    return sim3_from_components(inverse_scale, rotation_t, inverse_translation)


def sim3_exp(delta: torch.Tensor) -> torch.Tensor:
    """Exponential map for vectors ``[..., tx,ty,tz, rx,ry,rz, log_scale]``."""

    if delta.shape[-1] != 7:
        raise ValueError(f"Sim(3) tangent vector must end in 7, got {tuple(delta.shape)}")
    translation_tangent = delta[..., :3]
    omega = delta[..., 3:6]
    sigma = delta[..., 6]
    algebra = torch.zeros(*delta.shape[:-1], 4, 4, device=delta.device, dtype=delta.dtype)
    eye = torch.eye(3, device=delta.device, dtype=delta.dtype).expand(*delta.shape[:-1], 3, 3)
    algebra[..., :3, :3] = skew(omega) + sigma[..., None, None] * eye
    algebra[..., :3, 3] = translation_tangent
    return torch.matrix_exp(algebra)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    """Stable principal SO(3) logarithm over the complete [0, pi] range."""

    if rotation.shape[-2:] != (3, 3):
        raise ValueError(
            f"rotation must end in 3x3, got {tuple(rotation.shape)}"
        )
    eps = torch.finfo(rotation.dtype).eps
    m00, m01, m02 = rotation[..., 0, 0], rotation[..., 0, 1], rotation[..., 0, 2]
    m10, m11, m12 = rotation[..., 1, 0], rotation[..., 1, 1], rotation[..., 1, 2]
    m20, m21, m22 = rotation[..., 2, 0], rotation[..., 2, 1], rotation[..., 2, 2]
    q_abs = torch.stack(
        (
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ),
        dim=-1,
    ).clamp_min(eps).sqrt()
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    candidates = candidates / (2.0 * q_abs[..., :, None].clamp_min(eps))
    choice = q_abs.argmax(dim=-1)
    gather = choice[..., None, None].expand(*choice.shape, 1, 4)
    quaternion = candidates.gather(dim=-2, index=gather).squeeze(-2)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=eps)
    quaternion = torch.where(
        (quaternion[..., :1] < 0.0),
        -quaternion,
        quaternion,
    )
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    theta = 2.0 * torch.atan2(
        vector_norm,
        quaternion[..., 0].clamp_min(0.0),
    )
    axis = vector / vector_norm[..., None].clamp_min(eps)
    regular = theta[..., None] * axis
    return torch.where(
        (vector_norm < 1.0e-6)[..., None],
        2.0 * vector,
        regular,
    )


def sim3_log(transform: torch.Tensor) -> torch.Tensor:
    """Logarithm map matching :func:`sim3_exp`."""

    scale, rotation, translation = sim3_components(transform)
    sigma = scale.clamp_min(torch.finfo(scale.dtype).eps).log()
    omega = so3_log(rotation)
    eye = torch.eye(3, device=transform.device, dtype=transform.dtype).expand(*transform.shape[:-2], 3, 3)
    a = skew(omega) + sigma[..., None, None] * eye

    # exp([[A, I], [0, 0]]) has V(A) in its upper-right block, where
    # V(A)=integral_0^1 exp(tA) dt. This avoids a fragile A^{-1}(exp(A)-I)
    # near the identity.
    augmented = torch.zeros(*transform.shape[:-2], 6, 6, device=transform.device, dtype=transform.dtype)
    augmented[..., :3, :3] = a
    augmented[..., :3, 3:] = eye
    v = torch.matrix_exp(augmented)[..., :3, 3:]
    try:
        tangent_translation = torch.linalg.solve(v, translation[..., None]).squeeze(-1)
    except RuntimeError:
        tangent_translation = torch.linalg.lstsq(v, translation[..., None]).solution.squeeze(-1)
    return torch.cat([tangent_translation, omega, sigma[..., None]], dim=-1)


def apply_sim3(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    linear = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    return torch.einsum("...ij,...j->...i", linear, points) + translation


def apply_sim3_to_c2w(transform: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    """Apply a Sim(3) frame transform to an SE(3) camera-to-world pose."""

    scale, rotation, translation = sim3_components(transform)
    out = pose_c2w.clone()
    out[..., :3, :3] = rotation @ pose_c2w[..., :3, :3]
    out[..., :3, 3] = scale[..., None] * torch.einsum(
        "...ij,...j->...i", rotation, pose_c2w[..., :3, 3]
    ) + translation
    return out


def apply_sim3_to_pose(transform: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    """Backward-compatible alias for :func:`apply_sim3_to_c2w`."""

    return apply_sim3_to_c2w(transform, pose_c2w)


def rebase_c2w_to_sim3_anchor(transform: torch.Tensor, pose_c2w: torch.Tensor) -> torch.Tensor:
    """Express a global c2w pose in the local anchor of ``transform``.

    The returned SE(3) pose is the exact inverse of
    :func:`apply_sim3_to_c2w`, including the required ``1 / scale`` in its
    translation.
    """

    if pose_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"pose_c2w must end in 4x4, got {tuple(pose_c2w.shape)}")
    scale, rotation, translation = sim3_components(transform)
    rotation_t = rotation.transpose(-1, -2)
    out = pose_c2w.clone()
    out[..., :3, :3] = rotation_t @ pose_c2w[..., :3, :3]
    centered = pose_c2w[..., :3, 3] - translation
    out[..., :3, 3] = torch.einsum("...ij,...j->...i", rotation_t, centered) / scale[..., None]
    return out


def weighted_umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    *,
    allow_scale: bool = True,
) -> torch.Tensor:
    """Return the Sim(3) mapping ``source`` points onto ``target`` points."""

    if source.shape != target.shape or source.ndim != 2 or source.shape[-1] != 3:
        raise ValueError("source and target must both have shape Nx3")
    if weight is None:
        weight = torch.ones(source.shape[0], device=source.device, dtype=source.dtype)
    weight = weight.to(device=source.device, dtype=source.dtype).clamp_min(0.0)
    finite = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(weight)
        & (weight > 0)
    )
    source, target, weight = source[finite], target[finite], weight[finite]
    if source.shape[0] < 3:
        raise ValueError("weighted_umeyama requires at least three finite correspondences")
    weight = weight / weight.sum().clamp_min(1.0e-8)
    source_mean = (weight[:, None] * source).sum(dim=0)
    target_mean = (weight[:, None] * target).sum(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (weight[:, None] * source_centered).T @ target_centered
    u, singular, vh = torch.linalg.svd(covariance)
    rotation = vh.T @ u.T
    if float(torch.linalg.det(rotation).detach()) < 0.0:
        vh = vh.clone()
        vh[-1] *= -1.0
        rotation = vh.T @ u.T
    variance = (weight * source_centered.square().sum(dim=-1)).sum().clamp_min(1.0e-8)
    scale = singular.sum() / variance if allow_scale else source.new_tensor(1.0)
    translation = target_mean - scale * (rotation @ source_mean)
    return sim3_from_components(scale, rotation, translation)


def rotation_matrix_from_yaw(yaw: torch.Tensor | float, *, device=None, dtype=torch.float32) -> torch.Tensor:
    angle = torch.as_tensor(yaw, device=device, dtype=dtype)
    omega = torch.zeros(*angle.shape, 3, device=angle.device, dtype=angle.dtype)
    omega[..., 1] = angle
    return so3_exp(omega)
