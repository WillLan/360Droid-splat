"""Rotation-invariant spectral descriptors for feature fields on :math:`S^2`.

The implementation is intentionally dependency-free.  It evaluates an
orthonormal real spherical-harmonic basis with PyTorch, projects a multichannel
feature field into degree-wise coefficient matrices, and removes the unknown
SO(3) orientation with a cross-channel Gram matrix for every degree.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .spherical_erp import sample_erp_with_wrap, unit_ray_to_erp_pixel


SO3_SH_GRAM_DESCRIPTOR_VERSION = "so3_sh_gram_v1"


def fibonacci_sphere_directions(
    count: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic equal-area Fibonacci directions with shape ``Nx3``."""

    sample_count = int(count)
    if sample_count <= 0:
        raise ValueError(f"count must be positive, got {count!r}")
    index = torch.arange(sample_count, device=device, dtype=dtype)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * ((index + 0.5) / float(sample_count))
    longitude = torch.remainder(index * golden_angle, 2.0 * math.pi) - math.pi
    radius = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
    # Match geometry.spherical_erp: x=cos(lat)sin(lon), y=sin(lat),
    # z=cos(lat)cos(lon).
    return torch.stack(
        [radius * torch.sin(longitude), y, radius * torch.cos(longitude)],
        dim=-1,
    )


def real_spherical_harmonic_bands(
    max_degree: int,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Evaluate orthonormal real SH bands through ``max_degree``.

    Each returned tensor has shape ``... x (2*l+1)`` and is ordered by
    ``m=-l,...,+l``.  The polar axis follows the ERP convention's positive
    ``y`` direction.  The Condon-Shortley phase is included; degree-wise Gram
    invariance is independent of the chosen real-basis sign convention.
    """

    degree = int(max_degree)
    if degree < 0:
        raise ValueError(f"max_degree must be non-negative, got {max_degree!r}")
    if direction.shape[-1] != 3:
        raise ValueError(f"direction must end in three values, got {tuple(direction.shape)}")

    value = F.normalize(direction.float(), dim=-1, eps=1.0e-8)
    x, polar, z = value.unbind(dim=-1)
    longitude = torch.atan2(x, z)
    one_minus_polar_sq = (1.0 - polar.square()).clamp_min(0.0)

    # Store P_l^m(cos(theta)) for 0 <= m <= l <= degree.
    associated: dict[tuple[int, int], torch.Tensor] = {}
    p_mm = torch.ones_like(polar)
    for m in range(degree + 1):
        if m > 0:
            p_mm = -(2 * m - 1) * torch.sqrt(one_minus_polar_sq) * p_mm
        associated[(m, m)] = p_mm
        if m < degree:
            associated[(m + 1, m)] = (2 * m + 1) * polar * p_mm
        for ell in range(m + 2, degree + 1):
            associated[(ell, m)] = (
                (2 * ell - 1) * polar * associated[(ell - 1, m)]
                - (ell + m - 1) * associated[(ell - 2, m)]
            ) / float(ell - m)

    bands: list[torch.Tensor] = []
    root_two = math.sqrt(2.0)
    for ell in range(degree + 1):
        entries: list[torch.Tensor] = []
        for m in range(-ell, ell + 1):
            order = abs(m)
            normalization = math.exp(
                0.5
                * (
                    math.log(2 * ell + 1)
                    - math.log(4.0 * math.pi)
                    + math.lgamma(ell - order + 1)
                    - math.lgamma(ell + order + 1)
                )
            )
            base = associated[(ell, order)] * normalization
            if m < 0:
                entry = root_two * base * torch.sin(float(order) * longitude)
            elif m > 0:
                entry = root_two * base * torch.cos(float(order) * longitude)
            else:
                entry = base
            entries.append(entry)
        bands.append(torch.stack(entries, dim=-1))
    return tuple(bands)


def so3_sh_gram_descriptor_from_samples(
    feature: torch.Tensor,
    direction: torch.Tensor,
    *,
    max_degree: int = 6,
    weight: torch.Tensor | None = None,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Build SO(3)-invariant descriptors from equal-area spherical samples.

    Args:
        feature: ``... x N x C`` feature samples.
        direction: either ``N x 3`` directions shared by all leading batches,
            or ``... x N x 3`` directions matching ``feature``.
        weight: optional non-negative ``... x N`` reliability weights.

    Returns:
        ``... x D`` normalized descriptors where
        ``D=(max_degree+1)*(C*(C+1)/2+1)``.
    """

    if feature.ndim < 2:
        raise ValueError("feature must have shape ...xNxC")
    if direction.shape[-1] != 3 or int(direction.shape[-2]) != int(feature.shape[-2]):
        raise ValueError("direction must have shape Nx3 or ...xNx3 matching feature")
    samples = feature.float()
    bearings = direction.to(device=samples.device, dtype=torch.float32)
    if bearings.ndim == 2:
        bearings = bearings.reshape(*([1] * (samples.ndim - 2)), *bearings.shape)
    try:
        bearings = bearings.expand(*samples.shape[:-1], 3)
    except RuntimeError as exc:
        raise ValueError("direction leading dimensions are not broadcastable to feature") from exc

    finite = torch.isfinite(samples).all(dim=-1) & torch.isfinite(bearings).all(dim=-1)
    samples = F.normalize(torch.where(torch.isfinite(samples), samples, 0.0), dim=-1, eps=eps)
    if weight is None:
        reliability = torch.ones_like(samples[..., 0])
    else:
        reliability = weight.to(device=samples.device, dtype=torch.float32)
        try:
            reliability = reliability.expand_as(samples[..., 0])
        except RuntimeError as exc:
            raise ValueError("weight must be broadcastable to feature[..., 0]") from exc
    reliability = torch.where(
        finite & torch.isfinite(reliability) & (reliability > 0.0),
        reliability,
        torch.zeros_like(reliability),
    )
    weight_sum = reliability.sum(dim=-1, keepdim=True)
    if bool((weight_sum <= eps).any()):
        raise ValueError("SO(3) descriptor has no finite positive-weight spherical samples")
    reliability = reliability / weight_sum.clamp_min(eps)

    bands = real_spherical_harmonic_bands(max_degree, bearings)
    channel_count = int(samples.shape[-1])
    upper = torch.triu_indices(channel_count, channel_count, device=samples.device)
    parts: list[torch.Tensor] = []
    for basis in bands:
        # A_l: ... x C x (2l+1).  Equal-area quadrature contributes the
        # normalized reliability weight once, before the Gram contraction.
        coefficient = torch.einsum("...nc,...n,...nk->...ck", samples, reliability, basis)
        gram = coefficient @ coefficient.transpose(-1, -2)
        trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
        normalized = gram / trace.unsqueeze(-1).clamp_min(eps)
        parts.append(torch.cat([torch.log(trace.clamp_min(eps)), normalized[..., upper[0], upper[1]]], dim=-1))

    descriptor = torch.cat(parts, dim=-1)
    descriptor = descriptor.sign() * descriptor.abs().sqrt()
    return F.normalize(descriptor, dim=-1, eps=eps)


def build_so3_sh_gram_descriptor(
    features: torch.Tensor,
    *,
    max_degree: int = 6,
    num_samples: int = 2048,
    spatial_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build per-panorama SO(3)-invariant descriptors from ERP feature maps.

    ``features`` may be ``SxCxHxW`` or ``BxSxCxHxW``.  Reliability maps may be
    ``Sx1xHxW`` or ``BxSx1xHxW`` and are sampled at the same Fibonacci points.
    """

    value = features
    squeeze_batch = False
    if value.ndim == 4:
        value = value.unsqueeze(0)
        squeeze_batch = True
    if value.ndim != 5:
        raise ValueError("features must have shape SxCxHxW or BxSxCxHxW")
    batch, views, channels, height, width = (int(v) for v in value.shape)
    flat = value.float().reshape(batch * views, channels, height, width)
    directions = fibonacci_sphere_directions(
        num_samples,
        device=flat.device,
        dtype=torch.float32,
    )
    uv = unit_ray_to_erp_pixel(directions, height, width)
    sampled = sample_erp_with_wrap(flat, uv).reshape(batch, views, int(num_samples), channels)

    sampled_weight = None
    if spatial_weight is not None:
        reliability = spatial_weight
        if reliability.ndim == 4:
            reliability = reliability.unsqueeze(0)
        if reliability.ndim != 5 or tuple(int(v) for v in reliability.shape[:2]) != (batch, views):
            raise ValueError("spatial_weight must have shape Sx1xHxW or BxSx1xHxW")
        if int(reliability.shape[2]) != 1:
            raise ValueError("spatial_weight must contain exactly one channel")
        sampled_weight = sample_erp_with_wrap(
            reliability.float().reshape(batch * views, 1, *reliability.shape[-2:]),
            unit_ray_to_erp_pixel(directions, *reliability.shape[-2:]),
        )[..., 0].reshape(batch, views, int(num_samples))

    descriptor = so3_sh_gram_descriptor_from_samples(
        sampled,
        directions,
        max_degree=max_degree,
        weight=sampled_weight,
    )
    return descriptor[0] if squeeze_batch else descriptor
