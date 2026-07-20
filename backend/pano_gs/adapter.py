"""Adapter for PFGS360/gsplat360 equirectangular rendering.

The production path calls ``gsplat360.rasterization`` with
``camera_model="equirectangular"``.  A small CPU/PyTorch fallback is kept only
for tests and smoke runs where the CUDA extension is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import time
from typing import Any

import torch

from frontend.pano_droid.spherical_camera import bearing_to_erp_pixel, erp_pixel_to_bearing, pixel_grid
from geometry.pose import invert_c2w


@dataclass
class PanoRenderCamera:
    image_height: int
    image_width: int
    c2w: torch.Tensor

    @property
    def w2c(self) -> torch.Tensor:
        return invert_c2w(self.c2w)


RenderPackage = dict[str, torch.Tensor | None]
SH_C0 = 0.28209479177387814


def _optional_gsplat360(extra_roots: list[str] | None = None):
    try:
        from gsplat360 import rasterization
        from gsplat360.cuda import _backend as gsplat360_backend

        if gsplat360_backend._C is None:
            raise ImportError("gsplat360 CUDA extension is not loaded.")
        return rasterization
    except ModuleNotFoundError:
        pass

    for root in extra_roots or []:
        candidate = Path(root)
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        try:
            from gsplat360 import rasterization
            from gsplat360.cuda import _backend as gsplat360_backend

            if gsplat360_backend._C is None:
                raise ImportError("gsplat360 CUDA extension is not loaded.")
            return rasterization
        except ModuleNotFoundError:
            continue
    return None


def _identity_quats(n: int, *, device, dtype) -> torch.Tensor:
    quat = torch.zeros(n, 4, device=device, dtype=dtype)
    quat[:, 0] = 1.0
    return quat


def _blank_package(
    camera: PanoRenderCamera,
    gaussians,
    background: torch.Tensor,
    query_values: torch.Tensor | None = None,
) -> RenderPackage:
    H = int(camera.image_height)
    W = int(camera.image_width)
    total = int(gaussians.get_xyz.shape[0])
    bg = background.to(device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
    render = bg.view(3, 1, 1).expand(3, H, W).clone()
    depth = torch.zeros((1, H, W), device=render.device, dtype=render.dtype)
    alpha = torch.zeros((1, H, W), device=render.device, dtype=render.dtype)
    query_channels = 0 if query_values is None else int(query_values.shape[-1])
    return {
        "render": render,
        "gs_only": render,
        "sky_bg_only": torch.zeros_like(render),
        "sky_bg_alpha": torch.zeros_like(alpha),
        "depth": depth,
        "opacity": alpha,
        "alpha": alpha,
        "render_distort": None,
        "radii": torch.zeros(total, device=render.device, dtype=torch.int32),
        "n_touched": torch.zeros(total, device=render.device, dtype=torch.int32),
        "accum_visible": torch.zeros(total, device=render.device, dtype=render.dtype),
        "query_answers": (
            None
            if query_values is None
            else torch.zeros(total, query_channels, device=render.device, dtype=render.dtype)
        ),
        "accum_metric_counts": None,
        "viewspace_points": torch.zeros(total, 2, device=render.device, dtype=render.dtype),
        "visibility_filter": torch.zeros(total, device=render.device, dtype=torch.bool),
    }


def _blank_batched_package(
    cameras: list[PanoRenderCamera],
    gaussians,
    background: torch.Tensor,
) -> RenderPackage:
    if not cameras:
        raise ValueError("At least one camera is required.")
    count = len(cameras)
    height, width = int(cameras[0].image_height), int(cameras[0].image_width)
    total = int(gaussians.get_xyz.shape[0])
    device, dtype = gaussians.get_xyz.device, gaussians.get_xyz.dtype
    bg = background.to(device=device, dtype=dtype)
    if bg.ndim == 1:
        bg = bg.view(1, 3).expand(count, -1)
    render = bg[:, :, None, None].expand(count, 3, height, width).clone()
    depth = torch.zeros(count, 1, height, width, device=device, dtype=dtype)
    alpha = torch.zeros_like(depth)
    return {
        "render": render,
        "gs_only": render,
        "sky_bg_only": torch.zeros_like(render),
        "sky_bg_alpha": torch.zeros_like(alpha),
        "depth": depth,
        "opacity": alpha,
        "alpha": alpha,
        "render_distort": None,
        "radii": torch.zeros(count, total, device=device, dtype=torch.int32),
        "n_touched": torch.zeros(count, total, device=device, dtype=torch.int32),
        "accum_visible": None,
        "accum_metric_counts": None,
        "viewspace_points": torch.zeros(count, total, 2, device=device, dtype=dtype),
        "visibility_filter": torch.zeros(count, total, device=device, dtype=torch.bool),
    }


class PFGS360Renderer:
    """Render ``anchor_scaffold_panorama`` maps through gsplat360 when available."""

    def __init__(
        self,
        *,
        config: dict | None = None,
        extra_gsplat360_roots: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.config = config or {}
        self.extra_gsplat360_roots = extra_gsplat360_roots or []
        self.allow_fallback = bool(allow_fallback)

    def _training_cfg(self, gaussians) -> dict[str, Any]:
        cfg = getattr(gaussians, "config", None) or self.config
        return cfg.get("Training", {}) if isinstance(cfg, dict) else {}

    def _sync_for_profile(self, device: torch.device) -> None:
        cfg = {}
        if isinstance(self.config, dict):
            cfg = self.config.get("renderer", self.config.get("Renderer", {}))
        if not bool(cfg.get("profile_synchronize_cuda", False)):
            return
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    @staticmethod
    def _attach_profile(pkg: RenderPackage, profile: dict[str, float]) -> RenderPackage:
        out = dict(pkg)
        out.update(profile)
        return out

    def render(
        self,
        camera: PanoRenderCamera,
        gaussians,
        *,
        background: torch.Tensor | None = None,
        query_values: torch.Tensor | None = None,
    ) -> RenderPackage:
        total_start = time.perf_counter()
        profile: dict[str, float] = {
            "profile_renderer_materialize_sec": 0.0,
            "profile_renderer_rasterize_sec": 0.0,
            "profile_renderer_postprocess_sec": 0.0,
            "profile_renderer_skybox_sec": 0.0,
            "profile_renderer_total_sec": 0.0,
            "profile_renderer_materialized_gaussians": 0.0,
        }
        source_gaussians = gaussians
        materialized = None
        if hasattr(gaussians, "materialize"):
            section_start = time.perf_counter()
            materialized = gaussians.materialize(camera)
            gaussians = materialized
            self._sync_for_profile(gaussians.get_xyz.device)
            profile["profile_renderer_materialize_sec"] = float(time.perf_counter() - section_start)
            profile["profile_renderer_materialized_gaussians"] = float(int(gaussians.get_xyz.shape[0]))
        if background is None:
            background = torch.zeros(3, device=gaussians.get_xyz.device, dtype=gaussians.get_xyz.dtype)
        if int(gaussians.get_xyz.shape[0]) == 0:
            section_start = time.perf_counter()
            pkg = _blank_package(camera, gaussians, background, query_values)
            profile["profile_renderer_rasterize_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            pkg = self._postprocess_materialized(source_gaussians, materialized, pkg)
            profile["profile_renderer_postprocess_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            pkg = self._compose_skybox(camera, source_gaussians, pkg)
            profile["profile_renderer_skybox_sec"] = float(time.perf_counter() - section_start)
            profile["profile_renderer_total_sec"] = float(time.perf_counter() - total_start)
            return self._attach_profile(pkg, profile)

        rasterization = _optional_gsplat360(self.extra_gsplat360_roots)
        if rasterization is None:
            if query_values is not None:
                raise RuntimeError(
                    "Gaussian query attribution requires the gsplat360 CUDA renderer"
                )
            if not self.allow_fallback:
                raise ImportError(
                    "gsplat360 is unavailable. Install/build the PFGS360 gsplat360 package "
                    "or enable the explicit smoke fallback."
                )
            section_start = time.perf_counter()
            pkg = self._render_fallback(camera, gaussians, background)
            self._sync_for_profile(gaussians.get_xyz.device)
            profile["profile_renderer_rasterize_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            pkg = self._postprocess_materialized(source_gaussians, materialized, pkg)
            profile["profile_renderer_postprocess_sec"] = float(time.perf_counter() - section_start)
            section_start = time.perf_counter()
            pkg = self._compose_skybox(camera, source_gaussians, pkg)
            self._sync_for_profile(gaussians.get_xyz.device)
            profile["profile_renderer_skybox_sec"] = float(time.perf_counter() - section_start)
            profile["profile_renderer_total_sec"] = float(time.perf_counter() - total_start)
            return self._attach_profile(pkg, profile)

        section_start = time.perf_counter()
        pkg = self._render_gsplat360(
            rasterization,
            camera,
            gaussians,
            background,
            query_values=query_values,
        )
        self._sync_for_profile(gaussians.get_xyz.device)
        profile["profile_renderer_rasterize_sec"] = float(time.perf_counter() - section_start)
        section_start = time.perf_counter()
        pkg = self._postprocess_materialized(source_gaussians, materialized, pkg)
        profile["profile_renderer_postprocess_sec"] = float(time.perf_counter() - section_start)
        section_start = time.perf_counter()
        pkg = self._compose_skybox(camera, source_gaussians, pkg)
        self._sync_for_profile(gaussians.get_xyz.device)
        profile["profile_renderer_skybox_sec"] = float(time.perf_counter() - section_start)
        profile["profile_renderer_total_sec"] = float(time.perf_counter() - total_start)
        return self._attach_profile(pkg, profile)

    def render_cameras(
        self,
        cameras: list[PanoRenderCamera] | tuple[PanoRenderCamera, ...],
        gaussians,
        *,
        background: torch.Tensor | None = None,
    ) -> RenderPackage:
        """Rasterize shared geometry into all target cameras in one CUDA call."""

        camera_list = list(cameras)
        if not camera_list:
            raise ValueError("render_cameras requires at least one camera.")
        height, width = int(camera_list[0].image_height), int(camera_list[0].image_width)
        if any(
            (int(camera.image_height), int(camera.image_width)) != (height, width)
            for camera in camera_list
        ):
            raise ValueError("All batched cameras must share image dimensions.")
        total_start = time.perf_counter()
        profile: dict[str, float] = {
            "profile_renderer_materialize_sec": 0.0,
            "profile_renderer_rasterize_sec": 0.0,
            "profile_renderer_postprocess_sec": 0.0,
            "profile_renderer_skybox_sec": 0.0,
            "profile_renderer_total_sec": 0.0,
            "profile_renderer_materialized_gaussians": 0.0,
            "profile_renderer_batched_cameras": float(len(camera_list)),
        }
        if hasattr(gaussians, "materialize_batched"):
            section_start = time.perf_counter()
            gaussians = gaussians.materialize_batched(camera_list, batch_index=0)
            self._sync_for_profile(gaussians.get_xyz.device)
            profile["profile_renderer_materialize_sec"] = float(
                time.perf_counter() - section_start
            )
        profile["profile_renderer_materialized_gaussians"] = float(
            int(gaussians.get_xyz.shape[0])
        )
        if background is None:
            background = torch.zeros(
                len(camera_list),
                3,
                device=gaussians.get_xyz.device,
                dtype=gaussians.get_xyz.dtype,
            )
        if int(gaussians.get_xyz.shape[0]) == 0:
            package = _blank_batched_package(camera_list, gaussians, background)
            profile["profile_renderer_total_sec"] = float(time.perf_counter() - total_start)
            return self._attach_profile(package, profile)

        rasterization = _optional_gsplat360(self.extra_gsplat360_roots)
        if rasterization is None:
            raise ImportError(
                "Batched Stage 3 rendering requires the gsplat360 CUDA extension."
            )
        section_start = time.perf_counter()
        package = self._render_gsplat360_cameras(
            rasterization,
            camera_list,
            gaussians,
            background,
        )
        self._sync_for_profile(gaussians.get_xyz.device)
        profile["profile_renderer_rasterize_sec"] = float(
            time.perf_counter() - section_start
        )
        profile["profile_renderer_total_sec"] = float(time.perf_counter() - total_start)
        return self._attach_profile(package, profile)

    def _postprocess_materialized(self, source_gaussians, materialized, pkg: RenderPackage) -> RenderPackage:
        if materialized is None or not hasattr(source_gaussians, "postprocess_render_package"):
            return pkg
        return source_gaussians.postprocess_render_package(pkg, materialized)

    def _compose_skybox(self, camera: PanoRenderCamera, gaussians, pkg: RenderPackage) -> RenderPackage:
        if not bool(getattr(gaussians, "has_skybox", False)):
            return pkg
        rgb = pkg["render"]
        alpha = pkg.get("alpha")
        if not torch.is_tensor(rgb) or not torch.is_tensor(alpha):
            return pkg
        H = int(camera.image_height)
        W = int(camera.image_width)
        grid = pixel_grid(H, W, device=rgb.device, dtype=rgb.dtype).view(-1, 2)
        dirs_cam = erp_pixel_to_bearing(grid, H, W).to(device=rgb.device, dtype=rgb.dtype)
        c2w = camera.c2w.to(device=rgb.device, dtype=rgb.dtype)
        dirs_world = (c2w[:3, :3] @ dirs_cam.T).T.view(H, W, 3)
        sky_rgb = gaussians.sample_skybox(dirs_world).permute(2, 0, 1).contiguous().to(rgb)
        trans = (1.0 - alpha).clamp(0.0, 1.0)
        composed = rgb + trans * sky_rgb
        out = dict(pkg)
        out["render"] = composed.clamp(0.0, 1.0)
        out["gs_only"] = rgb
        out["sky_bg_only"] = sky_rgb
        out["sky_bg_alpha"] = trans
        return out

    def _render_gsplat360(
        self,
        rasterization,
        camera: PanoRenderCamera,
        gaussians,
        background: torch.Tensor,
        *,
        query_values: torch.Tensor | None = None,
    ) -> RenderPackage:
        H = int(camera.image_height)
        W = int(camera.image_width)
        xyz = gaussians.get_xyz
        device = xyz.device
        dtype = xyz.dtype
        training_cfg = self._training_cfg(gaussians)
        if bool(training_cfg.get("pfgs360_packed", False)):
            raise NotImplementedError("PFGS360 backend requires pfgs360_packed=False.")
        render_mode = str(training_cfg.get("pfgs360_render_mode", "RGB+ED"))
        if render_mode != "RGB+ED":
            raise NotImplementedError("PFGS360 backend expects pfgs360_render_mode='RGB+ED'.")

        viewmat = camera.w2c.to(device=device, dtype=dtype)
        K = torch.tensor(
            [[W / 2.0, 0.0, (W - 1.0) / 2.0], [0.0, H / 2.0, (H - 1.0) / 2.0], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        colors_sh = (
            gaussians.get_sh_coefficients
            if hasattr(gaussians, "get_sh_coefficients")
            else ((gaussians.get_features - 0.5) / SH_C0).unsqueeze(1)
        )
        if query_values is not None:
            query_values = query_values.to(device=device, dtype=torch.float32)
            if query_values.ndim == 3:
                query_values = query_values.unsqueeze(0)
            if (
                query_values.ndim != 4
                or int(query_values.shape[0]) != 1
                or tuple(query_values.shape[1:3]) != (H, W)
            ):
                raise ValueError(
                    "query_values must have shape HxWxK or 1xHxWxK for a single camera"
                )
        render, alpha, render_distort, info = rasterization(
            means=xyz,
            quats=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            opacities=gaussians.get_opacity.squeeze(-1),
            colors=colors_sh,
            viewmats=viewmat.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            packed=False,
            backgrounds=background.to(device=device, dtype=dtype).view(1, 3),
            near_plane=float(training_cfg.get("pfgs360_near_plane", 0.01)),
            far_plane=float(training_cfg.get("pfgs360_far_plane", 1.0e5)),
            radius_clip=float(training_cfg.get("pfgs360_radius_clip", 0.0)),
            render_mode=render_mode,
            sh_degree=int(getattr(gaussians, "active_sh_degree", 0)),
            sparse_grad=False,
            absgrad=bool(training_cfg.get("pfgs360_absgrad", True)),
            distloss=bool(training_cfg.get("pfgs360_distloss", False)),
            rasterize_mode=str(training_cfg.get("pfgs360_rasterize_mode", "antialiased")),
            camera_model="equirectangular",
            ret_visible=True,
            query_values=query_values,
        )
        rgb = render[0, ..., :3].permute(2, 0, 1).contiguous()
        depth = render[0, ..., 3:4].permute(2, 0, 1).contiguous()
        opacity = alpha[0].permute(2, 0, 1).contiguous()
        means2d = info["means2d"]
        if torch.is_tensor(means2d) and means2d.ndim == 3 and int(means2d.shape[0]) == 1:
            means2d = means2d[0]
        if means2d.requires_grad:
            means2d.retain_grad()
        radii = info["radii"][0]
        accumulated_visibility = info.get("accum_visible")
        if torch.is_tensor(accumulated_visibility):
            accumulated_visibility = accumulated_visibility[0]
            if tuple(accumulated_visibility.shape) != tuple(radii.shape):
                raise RuntimeError("gsplat360 accum_visible must have shape N.")
            visibility_filter = accumulated_visibility > 0
        else:
            visibility_filter = radii > 0
        query_answers = info.get("query_answers")
        if torch.is_tensor(query_answers):
            query_answers = query_answers[0]
            if int(query_answers.shape[0]) != int(xyz.shape[0]):
                raise RuntimeError("gsplat360 query_answers must have shape NxK.")
        n_touched = info.get("accum_times")
        if n_touched is not None:
            n_touched = n_touched[0].to(device=device, dtype=torch.int32)
        else:
            n_touched = visibility_filter.to(dtype=torch.int32)
        return {
            "render": rgb,
            "gs_only": rgb,
            "sky_bg_only": torch.zeros_like(rgb),
            "sky_bg_alpha": torch.zeros_like(opacity),
            "depth": depth,
            "opacity": opacity,
            "alpha": opacity,
            "render_distort": render_distort[0] if render_distort is not None else None,
            "radii": radii,
            "n_touched": n_touched,
            "accum_visible": accumulated_visibility,
            "query_answers": query_answers,
            "accum_metric_counts": None,
            "viewspace_points": means2d,
            "visibility_filter": visibility_filter,
        }

    def _render_gsplat360_cameras(
        self,
        rasterization,
        cameras: list[PanoRenderCamera],
        gaussians,
        background: torch.Tensor,
    ) -> RenderPackage:
        """Unpacked multi-camera PFGS360 call with shared opacity/geometry."""

        height, width = int(cameras[0].image_height), int(cameras[0].image_width)
        xyz = gaussians.get_xyz
        device, dtype = xyz.device, xyz.dtype
        training_cfg = self._training_cfg(gaussians)
        if bool(training_cfg.get("pfgs360_packed", False)):
            raise NotImplementedError(
                "The confirmed Stage 3 batched renderer uses pfgs360_packed=False."
            )
        render_mode = str(training_cfg.get("pfgs360_render_mode", "RGB+ED"))
        if render_mode != "RGB+ED":
            raise NotImplementedError(
                "PFGS360 batched rendering expects pfgs360_render_mode='RGB+ED'."
            )
        camera_count = len(cameras)
        viewmats = torch.stack(
            [camera.w2c.to(device=device, dtype=dtype) for camera in cameras], dim=0
        )
        intrinsic = torch.tensor(
            [
                [width / 2.0, 0.0, (width - 1.0) / 2.0],
                [0.0, height / 2.0, (height - 1.0) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        )
        intrinsics = intrinsic.unsqueeze(0).expand(camera_count, -1, -1)
        # Autocast may produce BF16 target-conditioned RGB even though Stage 3
        # materializes geometry in FP32. gsplat360's SH kernel requires colors
        # and view directions to share a scalar type.
        colors_sh = (
            gaussians.get_sh_coefficients
            if hasattr(gaussians, "get_sh_coefficients")
            else ((gaussians.get_features - 0.5) / SH_C0).unsqueeze(-2)
        ).to(device=device, dtype=dtype)
        if colors_sh.ndim == 3:
            if int(colors_sh.shape[0]) != int(xyz.shape[0]):
                raise ValueError("Shared Gaussian SH coefficients must have shape NxKx3.")
        elif colors_sh.ndim == 4:
            if tuple(colors_sh.shape[:2]) != (camera_count, int(xyz.shape[0])):
                raise ValueError("Batched Gaussian SH coefficients must have shape CxNxKx3.")
        else:
            raise ValueError("Gaussian SH coefficients must have shape NxKx3 or CxNxKx3.")
        backgrounds = background.to(device=device, dtype=dtype)
        if backgrounds.ndim == 1:
            backgrounds = backgrounds.view(1, 3).expand(camera_count, -1)
        if tuple(backgrounds.shape) != (camera_count, 3):
            raise ValueError("Batched backgrounds must have shape Cx3.")

        render, alpha, render_distort, info = rasterization(
            means=xyz,
            quats=gaussians.get_rotation,
            scales=gaussians.get_scaling,
            opacities=gaussians.get_opacity.squeeze(-1),
            colors=colors_sh,
            viewmats=viewmats,
            Ks=intrinsics,
            width=width,
            height=height,
            packed=False,
            backgrounds=backgrounds,
            near_plane=float(training_cfg.get("pfgs360_near_plane", 0.01)),
            far_plane=float(training_cfg.get("pfgs360_far_plane", 1.0e5)),
            radius_clip=float(training_cfg.get("pfgs360_radius_clip", 0.0)),
            render_mode=render_mode,
            sh_degree=int(getattr(gaussians, "active_sh_degree", 0)),
            sparse_grad=False,
            absgrad=bool(training_cfg.get("pfgs360_absgrad", True)),
            distloss=bool(training_cfg.get("pfgs360_distloss", False)),
            rasterize_mode=str(
                training_cfg.get("pfgs360_rasterize_mode", "antialiased")
            ),
            camera_model="equirectangular",
            ret_visible=True,
        )
        rgb = render[..., :3].permute(0, 3, 1, 2).contiguous()
        depth = render[..., 3:4].permute(0, 3, 1, 2).contiguous()
        opacity = alpha.permute(0, 3, 1, 2).contiguous()
        means2d = info["means2d"]
        if means2d.requires_grad:
            means2d.retain_grad()
        radii = info["radii"]
        if tuple(radii.shape) != (camera_count, int(xyz.shape[0])):
            raise RuntimeError(
                "Unpacked gsplat360 radii must have shape CxN for batched visibility."
            )
        accumulated_visibility = info.get("accum_visible")
        if torch.is_tensor(accumulated_visibility):
            if tuple(accumulated_visibility.shape) != tuple(radii.shape):
                raise RuntimeError("gsplat360 accum_visible must have shape CxN.")
            visibility_filter = accumulated_visibility > 0
        else:
            visibility_filter = radii > 0
        n_touched = info.get("accum_times")
        if n_touched is not None:
            n_touched = n_touched.to(device=device, dtype=torch.int32)
        else:
            n_touched = visibility_filter.to(dtype=torch.int32)
        return {
            "render": rgb,
            "gs_only": rgb,
            "sky_bg_only": torch.zeros_like(rgb),
            "sky_bg_alpha": torch.zeros_like(opacity),
            "depth": depth,
            "opacity": opacity,
            "alpha": opacity,
            "render_distort": render_distort,
            "radii": radii,
            "n_touched": n_touched,
            "accum_visible": accumulated_visibility,
            "accum_metric_counts": None,
            "viewspace_points": means2d,
            "visibility_filter": visibility_filter,
        }

    def _render_fallback(
        self,
        camera: PanoRenderCamera,
        gaussians,
        background: torch.Tensor,
    ) -> RenderPackage:
        """Naive ERP point renderer for tests.

        This fallback is intentionally not a replacement for gsplat360.  It
        only lets the project run integration tests without the CUDA extension.
        """
        H = int(camera.image_height)
        W = int(camera.image_width)
        xyz = gaussians.get_xyz.detach()
        device = xyz.device
        dtype = xyz.dtype
        render = background.to(device=device, dtype=dtype).view(3, 1, 1).expand(3, H, W).clone()
        depth = torch.zeros(1, H, W, device=device, dtype=dtype)
        alpha = torch.zeros(1, H, W, device=device, dtype=dtype)
        if xyz.numel() == 0:
            return _blank_package(camera, gaussians, background)

        w2c = camera.w2c.to(device=device, dtype=dtype)
        xyz_h = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=device, dtype=dtype)], dim=1)
        cam = (w2c @ xyz_h.T).T[:, :3]
        distance = torch.linalg.norm(cam, dim=-1)
        valid = torch.isfinite(cam).all(dim=-1) & torch.isfinite(distance) & (distance > 1.0e-4)
        if not bool(valid.any()):
            return _blank_package(camera, gaussians, background)

        cam_v = cam[valid]
        colors = gaussians.get_features.detach()[valid, :3].clamp(0.0, 1.0)
        opacity = gaussians.get_opacity.detach()[valid, 0].clamp(0.0, 1.0)
        pixels = bearing_to_erp_pixel(cam_v, H, W)
        ui = pixels[:, 0].round().long().remainder(W)
        vi = pixels[:, 1].round().long().clamp(0, H - 1)
        dist = distance[valid]
        for row in range(cam_v.shape[0]):
            a = opacity[row]
            u = int(ui[row])
            v = int(vi[row])
            render[:, v, u] = (1.0 - a) * render[:, v, u] + a * colors[row]
            depth[:, v, u] = dist[row]
            alpha[:, v, u] = torch.maximum(alpha[:, v, u], a)
        total = int(xyz.shape[0])
        radii = torch.zeros(total, device=device, dtype=torch.int32)
        radii[valid] = 1
        viewspace_points = torch.zeros(total, 2, device=device, dtype=dtype)
        viewspace_points[valid] = pixels
        return {
            "render": render,
            "gs_only": render,
            "sky_bg_only": torch.zeros_like(render),
            "sky_bg_alpha": torch.zeros_like(alpha),
            "depth": depth,
            "opacity": alpha,
            "alpha": alpha,
            "render_distort": None,
            "radii": radii,
            "n_touched": radii,
            "accum_metric_counts": None,
            "viewspace_points": viewspace_points,
            "visibility_filter": radii > 0,
        }
