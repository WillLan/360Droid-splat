"""PanoLOG-style distant Gaussian sky sphere.

The sphere is deliberately separate from the persistent scene Gaussian map.
It exposes the small renderer interface used by :class:`PFGS360Renderer`, but
has no owner, topology, depth-query, or lifecycle metadata.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from backend.pano_gs.adapter import SH_C0
from geometry.pose import invert_c2w
from geometry.spherical_erp import sample_erp_with_wrap, unit_ray_to_erp_pixel
from models.per_pixel_gaussian_observation import matrix_to_quaternion


class SkySphereCameraBoundaryError(RuntimeError):
    """Raised when a camera would leave the fixed distant sky sphere."""


def fibonacci_sphere(
    count: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic, approximately equal-area unit directions."""

    n = int(count)
    if n <= 0:
        raise ValueError("SkySphere.num_gaussians must be positive")
    index = torch.arange(n, device=device, dtype=dtype)
    y = 1.0 - 2.0 * (index + 0.5) / float(n)
    radial = torch.sqrt((1.0 - y.square()).clamp_min(0.0))
    angle = index * (math.pi * (3.0 - math.sqrt(5.0)))
    directions = torch.stack(
        [radial * torch.cos(angle), y, radial * torch.sin(angle)],
        dim=-1,
    )
    return torch.nn.functional.normalize(directions, dim=-1, eps=1.0e-12)


def tangent_quaternions(directions: torch.Tensor) -> torch.Tensor:
    """Orient local XY axes tangentially and local Z radially."""

    z_axis = torch.nn.functional.normalize(directions, dim=-1, eps=1.0e-12)
    up = torch.zeros_like(z_axis)
    up[:, 1] = 1.0
    fallback = torch.zeros_like(z_axis)
    fallback[:, 0] = 1.0
    near_pole = z_axis[:, 1].abs() > 0.95
    reference = torch.where(near_pole[:, None], fallback, up)
    x_axis = torch.nn.functional.normalize(
        torch.cross(reference, z_axis, dim=-1),
        dim=-1,
        eps=1.0e-12,
    )
    y_axis = torch.nn.functional.normalize(
        torch.cross(z_axis, x_axis, dim=-1),
        dim=-1,
        eps=1.0e-12,
    )
    rotation = torch.stack([x_axis, y_axis, z_axis], dim=-1)
    return matrix_to_quaternion(rotation)


class PanoLOGSkySphere(nn.Module):
    """Fixed-position Gaussian sphere with a finite online bootstrap stage."""

    def __init__(
        self,
        *,
        config: dict,
        scene_config: dict,
        device: torch.device | str,
    ) -> None:
        super().__init__()
        self.config = scene_config
        self.enabled = bool(config.get("enabled", False))
        self.num_gaussians = int(config.get("num_gaussians", 65536))
        self.bootstrap_chunks = max(1, int(config.get("bootstrap_chunks", 10)))
        self.optimize_all_chunks = bool(
            config.get("optimize_all_chunks", False)
        )
        self.optimize_steps_per_chunk = max(
            0, int(config.get("optimize_steps_per_chunk", 20))
        )
        self.sky_threshold = float(config.get("sky_threshold", 0.6))
        self.exclude_from_geometry = bool(
            config.get("exclude_from_geometry", True)
        )
        self.active_sh_degree = max(0, min(int(config.get("sh_degree", 3)), 3))
        self.max_sh_degree = self.active_sh_degree
        self.radius_scene_multiplier = float(
            config.get("radius_scene_multiplier", 8.0)
        )
        self.radius_camera_multiplier = float(
            config.get("radius_camera_multiplier", 16.0)
        )
        self.tangent_scale_factor = float(config.get("tangent_scale_factor", 0.65))
        self.radial_scale_factor = float(config.get("radial_scale_factor", 0.05))
        self.warn_camera_radius_ratio = float(
            config.get("warn_camera_radius_ratio", 0.25)
        )
        self.abort_camera_radius_ratio = float(
            config.get("abort_camera_radius_ratio", 0.8)
        )
        if str(config.get("initialization", "fibonacci")).strip().lower() != "fibonacci":
            raise ValueError("SkySphere.initialization must be 'fibonacci'")
        if not 0.0 < self.sky_threshold < 1.0:
            raise ValueError("SkySphere.sky_threshold must be in (0, 1)")
        if self.enabled and not self.exclude_from_geometry:
            raise ValueError(
                "SkySphere.exclude_from_geometry must remain true"
            )
        if self.radius_scene_multiplier <= 0.0 or self.radius_camera_multiplier <= 0.0:
            raise ValueError("SkySphere radius multipliers must be positive")
        if not (
            0.0
            <= self.warn_camera_radius_ratio
            < self.abort_camera_radius_ratio
            < 1.0
        ):
            raise ValueError("SkySphere camera radius ratios are invalid")

        target_device = torch.device(device)
        directions = (
            fibonacci_sphere(
                self.num_gaussians,
                device=target_device,
                dtype=torch.float32,
            )
            if self.enabled
            else torch.zeros(0, 3, device=target_device)
        )
        self.register_buffer("directions", directions)
        self.register_buffer("center", torch.zeros(3, device=target_device))
        self.register_buffer("radius", torch.zeros((), device=target_device))
        self.register_buffer(
            "chunks_completed",
            torch.zeros((), device=target_device, dtype=torch.int64),
        )
        self.register_buffer(
            "initialized_flag",
            torch.zeros((), device=target_device, dtype=torch.bool),
        )
        self.register_buffer(
            "frozen_flag",
            torch.zeros((), device=target_device, dtype=torch.bool),
        )

        rest_dim = max(0, (self.active_sh_degree + 1) ** 2 - 1)
        self.rotation = nn.Parameter(
            tangent_quaternions(directions)
            if self.enabled
            else torch.zeros(0, 4, device=target_device),
            requires_grad=False,
        )
        self.scaling = nn.Parameter(
            torch.zeros(self.num_gaussians if self.enabled else 0, 3, device=target_device),
            requires_grad=False,
        )
        self.opacity_logit = nn.Parameter(
            torch.full(
                (self.num_gaussians if self.enabled else 0, 1),
                -2.9444389791664403,
                device=target_device,
            ),
            requires_grad=False,
        )
        self.features = nn.Parameter(
            torch.zeros(
                self.num_gaussians if self.enabled else 0,
                3,
                device=target_device,
            ),
            requires_grad=False,
        )
        self.sh_rest = nn.Parameter(
            torch.zeros(
                self.num_gaussians if self.enabled else 0,
                rest_dim,
                3,
                device=target_device,
            ),
            requires_grad=False,
        )

    @property
    def initialized(self) -> bool:
        return bool(self.initialized_flag.item())

    @property
    def frozen(self) -> bool:
        return bool(self.frozen_flag.item()) and not self.optimize_all_chunks

    @property
    def get_xyz(self) -> torch.Tensor:
        return self.center[None] + self.radius * self.directions

    @property
    def get_rotation(self) -> torch.Tensor:
        if self.rotation.numel() == 0:
            return self.rotation
        return torch.nn.functional.normalize(self.rotation, dim=-1, eps=1.0e-12)

    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self.scaling)

    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity_logit)

    @property
    def get_features(self) -> torch.Tensor:
        return (0.5 + SH_C0 * self.features).clamp(0.0, 1.0)

    @property
    def get_sh_coefficients(self) -> torch.Tensor:
        dc = self.features.unsqueeze(1)
        if int(self.sh_rest.shape[1]) == 0:
            return dc
        return torch.cat([dc, self.sh_rest], dim=1)

    def parameter_groups(self, learning_rates: dict[str, float]) -> list[dict]:
        if not self.initialized or self.frozen:
            return []
        groups = [
            {
                "params": [self.features],
                "lr": float(learning_rates.get("feature_dc", 2.0e-3)),
                "name": "sky_features",
            },
            {
                "params": [self.opacity_logit],
                "lr": float(learning_rates.get("opacity", 1.0e-3)),
                "name": "sky_opacity",
            },
            {
                "params": [self.scaling],
                "lr": float(learning_rates.get("scaling", 1.0e-4)),
                "name": "sky_scaling",
            },
            {
                "params": [self.rotation],
                "lr": float(learning_rates.get("rotation", 1.0e-4)),
                "name": "sky_rotation",
            },
        ]
        if int(self.sh_rest.shape[1]) > 0:
            groups.insert(
                1,
                {
                    "params": [self.sh_rest],
                    "lr": float(learning_rates.get("sh_rest", 1.0e-4)),
                    "name": "sky_sh_rest",
                },
            )
        return groups

    def set_trainable(self, enabled: bool) -> None:
        active = bool(enabled and self.initialized and not self.frozen)
        for parameter in (
            self.rotation,
            self.scaling,
            self.opacity_logit,
            self.features,
            self.sh_rest,
        ):
            parameter.requires_grad_(active)

    @staticmethod
    def _normalize_probability(
        probability: torch.Tensor,
        *,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = probability.detach().to(device=device, dtype=dtype)
        while value.ndim > 2:
            value = value[0]
        if value.ndim != 2:
            raise ValueError("Sky probability must resolve to HxW")
        if tuple(value.shape) != (height, width):
            value = torch.nn.functional.interpolate(
                value.view(1, 1, *value.shape),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
        return value.clamp(0.0, 1.0)

    def _sample_observation(
        self,
        image: torch.Tensor,
        c2w: torch.Tensor,
        sky_probability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target = image.detach().to(device=self.center.device, dtype=torch.float32)
        if target.ndim == 4:
            target = target[0]
        if target.ndim != 3 or int(target.shape[0]) != 3:
            raise ValueError("SkySphere initialization image must be CxHxW")
        height, width = int(target.shape[-2]), int(target.shape[-1])
        pose = c2w.detach().to(device=target.device, dtype=target.dtype)
        world = self.get_xyz
        camera = (
            torch.cat(
                [
                    world,
                    torch.ones(world.shape[0], 1, device=world.device, dtype=world.dtype),
                ],
                dim=-1,
            )
            @ invert_c2w(pose).transpose(0, 1)
        )[:, :3]
        pixels = unit_ray_to_erp_pixel(camera, height, width)
        rgb = sample_erp_with_wrap(target, pixels)
        probability = self._normalize_probability(
            sky_probability,
            height=height,
            width=width,
            device=target.device,
            dtype=target.dtype,
        )
        sampled_probability = sample_erp_with_wrap(
            probability.unsqueeze(0),
            pixels,
        )[:, 0]
        return rgb, sampled_probability

    def _radius_candidate(
        self,
        scene_xyz: torch.Tensor,
        camera_centers: torch.Tensor,
    ) -> torch.Tensor:
        candidates: list[torch.Tensor] = []
        xyz = scene_xyz.detach().to(device=self.center.device, dtype=torch.float32)
        if xyz.numel() > 0:
            distance = torch.linalg.norm(xyz - self.center[None], dim=-1)
            valid = torch.isfinite(distance) & (distance > 0.0)
            if bool(valid.any()):
                extent = torch.quantile(distance[valid], 0.99)
                candidates.append(extent * self.radius_scene_multiplier)
        cameras = camera_centers.detach().to(device=self.center.device, dtype=torch.float32)
        if cameras.numel() > 0:
            distance = torch.linalg.norm(cameras - self.center[None], dim=-1)
            valid = torch.isfinite(distance)
            if bool(valid.any()):
                candidates.append(
                    distance[valid].amax().clamp_min(1.0e-3)
                    * self.radius_camera_multiplier
                )
        if not candidates:
            return torch.tensor(1.0, device=self.center.device)
        return torch.stack(candidates).amax().clamp_min(1.0e-3)

    @torch.no_grad()
    def initialize(
        self,
        *,
        observations: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        scene_xyz: torch.Tensor,
        camera_centers: torch.Tensor,
    ) -> bool:
        if not self.enabled or self.initialized:
            return self.initialized
        if not observations:
            return False
        root_pose = observations[0][1].detach().to(
            device=self.center.device,
            dtype=torch.float32,
        )
        self.center.copy_(root_pose[:3, 3])
        candidate = self._radius_candidate(scene_xyz, camera_centers)
        self.radius.copy_(candidate)
        angular = math.sqrt(4.0 * math.pi / float(self.num_gaussians))
        tangent = candidate * angular * self.tangent_scale_factor
        radial = tangent * self.radial_scale_factor
        actual_scale = torch.stack([tangent, tangent, radial]).view(1, 3)
        self.scaling.copy_(actual_scale.log().expand_as(self.scaling))

        rgb_accum = torch.zeros_like(self.features)
        weight_accum = torch.zeros(
            self.num_gaussians,
            1,
            device=self.center.device,
            dtype=torch.float32,
        )
        fallback_accum = torch.zeros_like(self.features)
        for image, pose, probability in observations:
            rgb, sampled_probability = self._sample_observation(
                image,
                pose,
                probability,
            )
            weight = sampled_probability[:, None]
            rgb_accum += rgb * weight
            weight_accum += weight
            fallback_accum += rgb
        fallback = fallback_accum / float(len(observations))
        rgb = torch.where(
            weight_accum > 1.0e-4,
            rgb_accum / weight_accum.clamp_min(1.0e-4),
            fallback,
        ).clamp(0.0, 1.0)
        mean_probability = (weight_accum / float(len(observations))).clamp(0.0, 1.0)
        opacity = (0.05 + 0.9 * mean_probability).clamp(0.01, 0.99)
        self.features.copy_((rgb - 0.5) / SH_C0)
        self.opacity_logit.copy_(torch.logit(opacity))
        self.initialized_flag.fill_(True)
        self.set_trainable(False)
        return True

    @torch.no_grad()
    def update_bootstrap_radius(
        self,
        *,
        scene_xyz: torch.Tensor,
        camera_centers: torch.Tensor,
    ) -> float:
        if (
            not self.initialized
            or self.frozen
            or int(self.chunks_completed.item()) >= self.bootstrap_chunks
        ):
            return float(self.radius.detach().cpu())
        candidate = self._radius_candidate(scene_xyz, camera_centers)
        if float(candidate) <= float(self.radius):
            return float(self.radius.detach().cpu())
        ratio = candidate / self.radius.clamp_min(1.0e-8)
        self.radius.copy_(candidate)
        self.scaling.add_(ratio.log())
        return float(candidate.detach().cpu())

    @torch.no_grad()
    def complete_chunk(self) -> bool:
        if not self.initialized:
            return False
        self.chunks_completed.add_(1)
        if (
            not self.optimize_all_chunks
            and int(self.chunks_completed.item()) >= self.bootstrap_chunks
        ):
            self.frozen_flag.fill_(True)
            self.set_trainable(False)
        elif self.optimize_all_chunks:
            self.frozen_flag.fill_(False)
        return self.frozen

    def camera_radius_ratio(self, c2w: torch.Tensor) -> float:
        if not self.initialized or float(self.radius) <= 0.0:
            return 0.0
        camera = c2w[:3, 3].detach().to(self.center)
        return float(
            (
                torch.linalg.norm(camera - self.center)
                / self.radius.clamp_min(1.0e-8)
            )
            .detach()
            .cpu()
        )

    def validate_camera(self, c2w: torch.Tensor) -> tuple[float, bool]:
        ratio = self.camera_radius_ratio(c2w)
        if ratio >= self.abort_camera_radius_ratio:
            raise SkySphereCameraBoundaryError(
                "Camera reached the frozen SkySphere boundary: "
                f"center_distance/radius={ratio:.4f} >= "
                f"{self.abort_camera_radius_ratio:.4f}"
            )
        return ratio, ratio >= self.warn_camera_radius_ratio

    def metadata(self) -> dict[str, object]:
        radius = float(self.radius.detach().cpu())
        count = int(self.directions.shape[0])
        return {
            "enabled": bool(self.enabled),
            "initialized": bool(self.initialized),
            "frozen": bool(self.frozen),
            "num_gaussians": count,
            "center": self.center.detach().cpu().tolist(),
            "radius": radius,
            "surface_density": (
                float(count / (4.0 * math.pi * radius * radius))
                if radius > 0.0
                else None
            ),
            "solid_angle_per_gaussian": (
                float(4.0 * math.pi / count) if count > 0 else None
            ),
            "chunks_completed": int(self.chunks_completed.item()),
            "bootstrap_chunks": int(self.bootstrap_chunks),
            "radius_bootstrap_complete": bool(
                int(self.chunks_completed.item()) >= self.bootstrap_chunks
            ),
            "optimize_all_chunks": bool(self.optimize_all_chunks),
            "frozen_chunk": (
                int(self.bootstrap_chunks) if self.frozen else None
            ),
            "sky_threshold": float(self.sky_threshold),
            "exclude_from_geometry": bool(self.exclude_from_geometry),
        }

    def save_ply(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self.get_xyz.detach().cpu().float().numpy()
        n = int(xyz.shape[0])
        normals = self.directions.detach().cpu().float().numpy()
        coefficients = self.get_sh_coefficients.detach().cpu().float()
        f_dc = coefficients[:, 0].numpy()
        rest = coefficients[:, 1:16]
        if int(rest.shape[1]) < 15:
            padded = torch.zeros(n, 15, 3, dtype=rest.dtype)
            padded[:, : int(rest.shape[1])] = rest
            rest = padded
        f_rest = rest.permute(0, 2, 1).reshape(n, 45).numpy()
        opacity = self.opacity_logit.detach().cpu().float().numpy()
        scale = self.scaling.detach().cpu().float().numpy()
        rotation = self.get_rotation.detach().cpu().float().numpy()
        attributes = [
            "x",
            "y",
            "z",
            "nx",
            "ny",
            "nz",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            *(f"f_rest_{index}" for index in range(45)),
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ]
        dtype = np.dtype([(name, "<f4") for name in attributes])
        elements = np.empty(n, dtype=dtype)
        values = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacity, scale, rotation),
            axis=1,
        ).astype(np.float32, copy=False)
        for index, name in enumerate(attributes):
            elements[name] = values[:, index]
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            *(f"property float {name}" for name in attributes),
            "end_header",
        ]
        with open(path, "wb") as file:
            file.write(("\n".join(header) + "\n").encode("ascii"))
            elements.tofile(file)
        return str(path)
