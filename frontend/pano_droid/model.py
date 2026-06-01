"""DROID-style PanoDROID-MVP frontend model."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .correlation import SphericalCorrBlock, coords_grid
from .encoders import BasicEncoder, ContextEncoder
from .sphere_gru import SphereConvGRU
from .spherical_ba import se3_exp
from .spherical_camera import (
    bearing_to_erp_pixel,
    erp_pixel_to_bearing,
    pixel_grid,
    seam_aware_delta,
)


def _resize_like(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


def _resize_depth_like(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


class GraphAggregator(nn.Module):
    """Aggregate edge hidden states into per-source-frame damping."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.damping = nn.Sequential(
            nn.Conv2d(hidden_dim, max(1, hidden_dim // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, hidden_dim // 2), 1, 3, padding=1),
        )

    def forward(self, edge_hidden: torch.Tensor, ii: torch.Tensor, num_frames: int) -> torch.Tensor:
        B, E, C, H, W = edge_hidden.shape
        frame_hidden = edge_hidden.new_zeros(B, int(num_frames), C, H, W)
        counts = edge_hidden.new_zeros(int(num_frames))
        frame_hidden.index_add_(1, ii, edge_hidden)
        counts.index_add_(0, ii, torch.ones_like(ii, dtype=edge_hidden.dtype))
        frame_hidden = frame_hidden / counts.clamp_min(1.0).view(1, -1, 1, 1, 1)
        x = frame_hidden.reshape(B * int(num_frames), C, H, W)
        x = F.silu(self.conv1(x), inplace=True)
        x = F.silu(self.conv2(x), inplace=True)
        damping = F.softplus(self.damping(x)) + 1e-4
        return damping.view(B, int(num_frames), 1, H, W)


class PanoDroidModel(nn.Module):
    """Trainable DROID-style dense PanoDROID frontend network."""

    def __init__(
        self,
        *,
        feature_dim: int = 96,
        context_dim: int = 96,
        hidden_dim: int = 96,
        encoder_base_dim: int | None = None,
        feature_stride: int = 8,
        corr_levels: int = 4,
        corr_radius: int = 3,
        gru_kernel_size: int = 3,
        update_iters: int = 4,
        pose_scale: float = 0.02,
        max_corr_elements: int = 80_000_000,
        use_spherical_corr: bool = True,
        **unused,
    ) -> None:
        super().__init__()
        if int(feature_stride) != 8:
            raise ValueError("PanoDroidModel currently supports feature_stride=8.")
        self.feature_dim = int(feature_dim)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.feature_stride = int(feature_stride)
        self.corr_levels = int(corr_levels)
        self.corr_radius = int(corr_radius)
        self.update_iters = int(update_iters)
        self.pose_scale = float(pose_scale)
        self.max_corr_elements = int(max_corr_elements)
        self.use_spherical_corr = bool(use_spherical_corr)

        self.fnet = BasicEncoder(
            input_dim=3,
            output_dim=self.feature_dim,
            base_dim=encoder_base_dim,
        )
        self.cnet = ContextEncoder(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            context_dim=self.context_dim,
            base_dim=encoder_base_dim,
        )

        corr_dim = self.corr_levels * (2 * self.corr_radius + 1) ** 2
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.context_dim + corr_dim + 4, self.hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.update_block = SphereConvGRU(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            kernel_size=gru_kernel_size,
        )
        self.delta_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(self.hidden_dim, 2, 3, padding=1),
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, max(1, self.hidden_dim // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, self.hidden_dim // 2), 1, 3, padding=1),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, max(1, self.hidden_dim // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, self.hidden_dim // 2), 1, 3, padding=1),
        )
        self.damping_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, max(1, self.hidden_dim // 2), 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(1, self.hidden_dim // 2), 1, 3, padding=1),
        )
        self.graph_agg = GraphAggregator(self.hidden_dim)
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_dim, 6),
        )
        self.keyframe_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, max(1, self.hidden_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Linear(max(1, self.hidden_dim // 2), 1),
        )

        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.zeros_(self.pose_head[-1].weight)
        nn.init.zeros_(self.pose_head[-1].bias)
        nn.init.constant_(self.depth_head[-1].bias, -1.5)

    @staticmethod
    def _split_inputs(
        image0: torch.Tensor,
        image1: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image1 is None:
            if image0.ndim != 5 or image0.shape[1] < 2:
                raise ValueError(
                    "Pass image0/image1 as BxCxHxW tensors or images as BxTxCxHxW."
                )
            image1 = image0[:, 1]
            image0 = image0[:, 0]
        if image0.ndim != 4 or image1.ndim != 4:
            raise ValueError("Images must be BxCxHxW tensors.")
        if image0.shape != image1.shape:
            raise ValueError(f"Image shape mismatch: {tuple(image0.shape)} vs {tuple(image1.shape)}")
        return image0.float().clamp(0.0, 1.0), image1.float().clamp(0.0, 1.0)

    def _pad_to_stride(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, H, W = x.shape
        pad_h = (self.feature_stride - H % self.feature_stride) % self.feature_stride
        pad_w = (self.feature_stride - W % self.feature_stride) % self.feature_stride
        if pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, 0), mode="circular")
        if pad_h > 0:
            x = F.pad(x, (0, 0, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    def _make_corr_block(self, fmap0: torch.Tensor, fmap1: torch.Tensor) -> SphericalCorrBlock:
        return SphericalCorrBlock(
            fmap0,
            fmap1,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
            latitude_scale=self.use_spherical_corr,
        )

    @staticmethod
    def _edge_tensors(edges: list[tuple[int, int]], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if not edges:
            raise ValueError("forward_graph requires at least one edge.")
        ii, jj = zip(*edges)
        return (
            torch.as_tensor(ii, dtype=torch.long, device=device),
            torch.as_tensor(jj, dtype=torch.long, device=device),
        )

    @staticmethod
    def _initial_poses(
        poses_c2w: Optional[torch.Tensor],
        *,
        batch: int,
        frames: int,
        fixed_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if poses_c2w is None:
            eye = torch.eye(4, device=device, dtype=dtype)
            return eye.view(1, 1, 4, 4).expand(batch, frames, -1, -1).clone()
        init = poses_c2w.to(device=device, dtype=dtype).clone()
        fixed = max(1, min(int(fixed_frames), frames))
        anchor = fixed - 1
        if fixed < frames:
            init[:, fixed:] = init[:, anchor : anchor + 1].expand(-1, frames - fixed, -1, -1)
        return init

    @staticmethod
    def _downsample_inverse_depth(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        B, N = x.shape[:2]
        low = _resize_depth_like(x.reshape(B * N, 1, x.shape[-2], x.shape[-1]), size)
        return low.view(B, N, 1, size[0], size[1])

    @staticmethod
    def _relative_from_c2w(poses_c2w: torch.Tensor, ii: torch.Tensor, jj: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(poses_c2w[:, jj]) @ poses_c2w[:, ii]

    @staticmethod
    def _project_edges(
        poses_c2w: torch.Tensor,
        inverse_depth: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        height: int,
        width: int,
        pixels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = poses_c2w.shape[0]
        E = int(ii.numel())
        if pixels is None:
            pixels = pixel_grid(height, width, device=poses_c2w.device, dtype=poses_c2w.dtype)
        else:
            pixels = pixels.to(device=poses_c2w.device, dtype=poses_c2w.dtype)
        Hs, Ws = int(pixels.shape[0]), int(pixels.shape[1])
        inv_map = inverse_depth[:, ii].reshape(B * E, 1, height, width)
        norm_x = 2.0 * (pixels[..., 0] - 0.5) / max(width - 1, 1) - 1.0
        norm_y = 2.0 * (pixels[..., 1] - 0.5) / max(height - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).view(1, Hs, Ws, 2).expand(B * E, -1, -1, -1)
        inv_src = F.grid_sample(
            inv_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).view(B, E, Hs, Ws)
        p = pixels.view(1, 1, Hs, Ws, 2).expand(B, E, -1, -1, -1)
        bearing_i = erp_pixel_to_bearing(p.reshape(B * E, Hs * Ws, 2), height, width)
        bearing_i = bearing_i.view(B, E, Hs, Ws, 3)
        xyz_i = bearing_i / inv_src.clamp_min(1e-6).unsqueeze(-1)
        T_ji = torch.linalg.inv(poses_c2w[:, jj]) @ poses_c2w[:, ii]
        xyz_j = torch.einsum("beij,behwj->behwi", T_ji[..., :3, :3], xyz_i) + T_ji[
            ..., :3, 3
        ].view(B, E, 1, 1, 3)
        bearing_j = F.normalize(xyz_j, dim=-1, eps=1e-12)
        target_pixels = bearing_to_erp_pixel(bearing_j.reshape(B * E, Hs * Ws, 3), height, width)
        return target_pixels.view(B, E, Hs, Ws, 2)

    def _ba_refine(
        self,
        poses_c2w: torch.Tensor,
        inverse_depth: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor,
        damping: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
        *,
        fixed_frames: int,
        iters: int,
        sample_stride: int,
        lr: float = 5e-2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _, H, W = inverse_depth.shape
        stride = max(1, int(sample_stride))
        pixels = pixel_grid(H, W, device=target.device, dtype=target.dtype)[::stride, ::stride]
        target_s = target[:, :, ::stride, ::stride]
        weight_s = weight[:, :, :, ::stride, ::stride].squeeze(2)
        pose_mask = torch.ones(B, N, 1, device=poses_c2w.device, dtype=poses_c2w.dtype)
        pose_mask[:, : max(1, min(int(fixed_frames), N))] = 0.0

        with torch.enable_grad():
            base_poses = poses_c2w
            xi = torch.zeros(B, N, 6, device=poses_c2w.device, dtype=poses_c2w.dtype, requires_grad=True)
            log_inv = inverse_depth.clamp_min(1e-6).log()
            log_inv.requires_grad_(True)
            for _ in range(max(0, int(iters))):
                xi_used = xi * pose_mask
                cur_poses = se3_exp(xi_used) @ base_poses
                cur_inv = log_inv.exp().clamp_min(1e-6)
                coords = self._project_edges(cur_poses, cur_inv, ii, jj, height=H, width=W, pixels=pixels)
                residual = seam_aware_delta(coords, target_s, W)
                residual_norm = torch.sqrt((residual * residual).sum(dim=-1) + 1e-6)
                loss = (residual_norm * weight_s).sum() / weight_s.sum().clamp_min(1e-6)
                grad_xi, grad_log_inv = torch.autograd.grad(
                    loss,
                    (xi, log_inv),
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )
                frame_damp = 1.0 + damping.mean(dim=(-1, -2, -3), keepdim=False).view(B, N, 1)
                if grad_xi is not None:
                    xi = xi - float(lr) * grad_xi * pose_mask / frame_damp
                if grad_log_inv is not None:
                    depth_damp = 1.0 + damping
                    log_inv = log_inv - float(lr) * grad_log_inv / depth_damp

            cur_poses = se3_exp(xi * pose_mask) @ base_poses
            cur_inv = log_inv.exp().clamp_min(1e-6)
            coords_full = self._project_edges(cur_poses, cur_inv, ii, jj, height=H, width=W)
            residual_full = seam_aware_delta(coords_full, target, W)
        return cur_poses, cur_inv, residual_full

    def forward(
        self,
        image0: torch.Tensor,
        image1: Optional[torch.Tensor] = None,
        *,
        edges: Optional[list[tuple[int, int]]] = None,
        num_updates: Optional[int] = None,
        poses_c2w: Optional[torch.Tensor] = None,
        init_poses_c2w: Optional[torch.Tensor] = None,
        init_inverse_depth: Optional[torch.Tensor] = None,
        ba_iters_per_update: int = 2,
        fixed_frames: int = 2,
        ba_sample_stride: int = 1,
    ) -> dict[str, torch.Tensor] | dict[str, torch.Tensor | list[tuple[int, int]]]:
        if edges is not None or image0.ndim == 5:
            if edges is None:
                raise ValueError("Graph forward requires an explicit edge list.")
            return self.forward_graph(
                image0,
                edges=edges,
                num_updates=num_updates,
                poses_c2w=poses_c2w,
                init_poses_c2w=init_poses_c2w,
                init_inverse_depth=init_inverse_depth,
                ba_iters_per_update=ba_iters_per_update,
                fixed_frames=fixed_frames,
                ba_sample_stride=ba_sample_stride,
            )

        image0, image1 = self._split_inputs(image0, image1)
        B, _, H0, W0 = image0.shape
        image0_pad, _ = self._pad_to_stride(image0)
        image1_pad, _ = self._pad_to_stride(image1)
        _, _, Hp, Wp = image0_pad.shape
        iters = int(num_updates or self.update_iters)

        fmap0 = self.fnet(image0_pad)
        fmap1 = self.fnet(image1_pad)
        h, context = self.cnet(image0_pad)
        Hf, Wf = fmap0.shape[-2:]
        corr_block = self._make_corr_block(fmap0, fmap1)
        coords0 = coords_grid(B, Hf, Wf, device=image0.device, dtype=image0.dtype)
        coords1 = coords0.clone()

        for _ in range(iters):
            corr = corr_block(coords1)
            flow = coords1 - coords0
            motion = torch.cat([flow, torch.zeros_like(flow)], dim=1)
            gru_in = torch.cat([context, corr, motion], dim=1)
            gru_in = self.input_proj(gru_in)
            h = self.update_block(h, gru_in)
            coords1 = coords1 + self.delta_head(h)

        flow_low = coords1 - coords0
        flow_full = _resize_like(flow_low, (Hp, Wp))
        flow_full[:, 0] *= float(Wp) / max(float(Wf), 1.0)
        flow_full[:, 1] *= float(Hp) / max(float(Hf), 1.0)
        flow_full = flow_full[..., :H0, :W0]

        confidence = torch.sigmoid(_resize_like(self.conf_head(h), (Hp, Wp)))[..., :H0, :W0]
        inverse_depth = F.softplus(_resize_like(self.depth_head(h), (Hp, Wp)))[..., :H0, :W0] + 1e-4
        damping = F.softplus(_resize_like(self.damping_head(h), (Hp, Wp)))[..., :H0, :W0] + 1e-4
        pose_delta = self.pose_scale * torch.tanh(self.pose_head(h))
        relative_pose = se3_exp(pose_delta)
        flow_mean = flow_full.abs().mean(dim=(1, 2, 3), keepdim=False).view(B, 1)
        conf_mean = confidence.mean(dim=(1, 2, 3), keepdim=False).view(B, 1)
        depth_var = inverse_depth.var(dim=(1, 2, 3), keepdim=False).view(B, 1)
        key_in = torch.cat(
            [h.mean(dim=(2, 3)), flow_mean, conf_mean, depth_var.clamp_max(10.0)], dim=1
        )
        keyframe_score = torch.sigmoid(self.keyframe_head(key_in)).squeeze(-1)

        return {
            "spherical_flow": flow_full,
            "confidence": confidence,
            "depth_confidence": confidence,
            "inverse_depth": inverse_depth,
            "damping": damping,
            "pose_delta": pose_delta,
            "relative_pose": relative_pose,
            "keyframe_score": keyframe_score,
            "hidden": h,
            "flow_low": flow_low,
        }

    def forward_graph(
        self,
        images: torch.Tensor,
        *,
        edges: list[tuple[int, int]],
        num_updates: Optional[int] = None,
        poses_c2w: Optional[torch.Tensor] = None,
        init_poses_c2w: Optional[torch.Tensor] = None,
        init_inverse_depth: Optional[torch.Tensor] = None,
        ba_iters_per_update: int = 2,
        fixed_frames: int = 2,
        ba_sample_stride: int = 1,
    ) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        """Run DROID-style recurrent graph updates and spherical BA."""
        return self.forward_graph_droid_style(
            images,
            edges=edges,
            num_updates=num_updates,
            poses_c2w=poses_c2w,
            init_poses_c2w=init_poses_c2w,
            init_inverse_depth=init_inverse_depth,
            ba_iters_per_update=ba_iters_per_update,
            fixed_frames=fixed_frames,
            ba_sample_stride=ba_sample_stride,
        )

    def forward_graph_droid_style(
        self,
        images: torch.Tensor,
        *,
        edges: list[tuple[int, int]],
        num_updates: Optional[int] = None,
        poses_c2w: Optional[torch.Tensor] = None,
        init_poses_c2w: Optional[torch.Tensor] = None,
        init_inverse_depth: Optional[torch.Tensor] = None,
        ba_iters_per_update: int = 2,
        fixed_frames: int = 2,
        ba_sample_stride: int = 1,
    ) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        if images.ndim != 5:
            raise ValueError(f"Expected images as BxNxCxHxW, got {tuple(images.shape)}")
        B, N, C, H0, W0 = images.shape
        ii, jj = self._edge_tensors(edges, device=images.device)
        if bool(((ii < 0) | (ii >= N) | (jj < 0) | (jj >= N)).any()):
            raise IndexError(f"Graph edges are outside sequence length {N}.")
        E = int(ii.numel())
        iters = int(num_updates or self.update_iters)

        flat = images.float().clamp(0.0, 1.0).reshape(B * N, C, H0, W0)
        flat_pad, _ = self._pad_to_stride(flat)
        _, _, Hp, Wp = flat_pad.shape
        fmaps = self.fnet(flat_pad)
        hidden, context = self.cnet(flat_pad)
        Hf, Wf = fmaps.shape[-2:]
        fmaps = fmaps.view(B, N, self.feature_dim, Hf, Wf)
        hidden = hidden.view(B, N, self.hidden_dim, Hf, Wf)
        context = context.view(B, N, self.context_dim, Hf, Wf)

        inv_low_init = F.softplus(self.depth_head(hidden.reshape(B * N, self.hidden_dim, Hf, Wf)))
        inv_low_init = inv_low_init.view(B, N, 1, Hf, Wf) + 1e-4
        if init_inverse_depth is not None:
            inv_depth_state = self._downsample_inverse_depth(
                init_inverse_depth.to(device=images.device, dtype=images.dtype), (Hf, Wf)
            )
        else:
            inv_depth_state = inv_low_init

        if init_poses_c2w is not None:
            poses_state = init_poses_c2w.to(device=images.device, dtype=images.dtype)
        else:
            poses_state = self._initial_poses(
                poses_c2w,
                batch=B,
                frames=N,
                fixed_frames=fixed_frames,
                device=images.device,
                dtype=images.dtype,
            )

        f0 = fmaps[:, ii].reshape(B * E, self.feature_dim, Hf, Wf)
        f1 = fmaps[:, jj].reshape(B * E, self.feature_dim, Hf, Wf)
        corr_block = self._make_corr_block(f0, f1)
        edge_hidden = hidden[:, ii].clone()
        edge_context = context[:, ii]
        coords0 = coords_grid(B * E, Hf, Wf, device=images.device, dtype=images.dtype)
        coords0_hw = coords0.permute(0, 2, 3, 1).view(B, E, Hf, Wf, 2)
        coords1 = self._project_edges(poses_state, inv_depth_state, ii, jj, height=Hf, width=Wf)
        target = coords1.clone()

        pose_steps = []
        inv_steps = []
        residual_steps = []
        target_steps = []
        weight_steps = []
        damping_steps = []

        for _ in range(iters):
            poses_state = poses_state.detach()
            coords1 = coords1.detach()
            target = target.detach()
            corr = corr_block(coords1.reshape(B * E, Hf, Wf, 2).permute(0, 3, 1, 2))
            flow = seam_aware_delta(coords0_hw, coords1, Wf).permute(0, 1, 4, 2, 3)
            resd = seam_aware_delta(coords1, target, Wf).permute(0, 1, 4, 2, 3)
            motion = torch.cat([flow, resd], dim=2).reshape(B * E, 4, Hf, Wf).clamp(-64.0, 64.0)
            gru_in = torch.cat(
                [
                    edge_context.reshape(B * E, self.context_dim, Hf, Wf),
                    corr,
                    motion,
                ],
                dim=1,
            )
            gru_in = self.input_proj(gru_in)
            edge_hidden_flat = edge_hidden.reshape(B * E, self.hidden_dim, Hf, Wf)
            edge_hidden_flat = self.update_block(edge_hidden_flat, gru_in)
            edge_hidden = edge_hidden_flat.view(B, E, self.hidden_dim, Hf, Wf)
            delta = self.delta_head(edge_hidden_flat).view(B, E, 2, Hf, Wf).permute(0, 1, 3, 4, 2)
            edge_weight = torch.sigmoid(self.conf_head(edge_hidden_flat)).view(B, E, 1, Hf, Wf)
            damping = self.graph_agg(edge_hidden, ii, N)
            target = coords1 + delta
            poses_state, inv_depth_state, residual = self._ba_refine(
                poses_state,
                inv_depth_state,
                target,
                edge_weight,
                damping,
                ii,
                jj,
                fixed_frames=fixed_frames,
                iters=ba_iters_per_update,
                sample_stride=ba_sample_stride,
            )
            coords1 = self._project_edges(poses_state, inv_depth_state, ii, jj, height=Hf, width=Wf)
            pose_steps.append(poses_state)
            inv_steps.append(inv_depth_state)
            residual_steps.append(residual)
            target_steps.append(target)
            weight_steps.append(edge_weight)
            damping_steps.append(damping)

        if not pose_steps:
            damping = self.graph_agg(edge_hidden, ii, N)
            edge_weight = torch.sigmoid(self.conf_head(edge_hidden.reshape(B * E, self.hidden_dim, Hf, Wf))).view(B, E, 1, Hf, Wf)
            residual = seam_aware_delta(coords1, target, Wf)
            pose_steps.append(poses_state)
            inv_steps.append(inv_depth_state)
            residual_steps.append(residual)
            target_steps.append(target)
            weight_steps.append(edge_weight)
            damping_steps.append(damping)

        poses_stack = torch.stack(pose_steps, dim=1)
        inv_stack = torch.stack(inv_steps, dim=1)
        residual_stack = torch.stack(residual_steps, dim=1)
        target_stack = torch.stack(target_steps, dim=1)
        weight_stack = torch.stack(weight_steps, dim=1)
        damping_stack = torch.stack(damping_steps, dim=1)
        final_poses = poses_stack[:, -1]
        final_inv = inv_stack[:, -1]
        final_weight = weight_stack[:, -1]
        final_damping = damping_stack[:, -1]
        final_coords = self._project_edges(final_poses, final_inv, ii, jj, height=Hf, width=Wf)
        flow_low_hw = seam_aware_delta(coords0_hw, final_coords, Wf)
        flow_low = flow_low_hw.permute(0, 1, 4, 2, 3).contiguous()
        flow_full = _resize_like(flow_low.reshape(B * E, 2, Hf, Wf), (Hp, Wp))
        flow_full[:, 0] *= float(Wp) / max(float(Wf), 1.0)
        flow_full[:, 1] *= float(Hp) / max(float(Hf), 1.0)
        flow_full = flow_full[..., :H0, :W0].view(B, E, 2, H0, W0)

        inv_edge_full = _resize_depth_like(final_inv[:, ii].reshape(B * E, 1, Hf, Wf), (Hp, Wp))
        inv_edge_full = inv_edge_full[..., :H0, :W0].view(B, E, 1, H0, W0)
        conf_full = _resize_depth_like(final_weight.reshape(B * E, 1, Hf, Wf), (Hp, Wp))
        conf_full = conf_full[..., :H0, :W0].view(B, E, 1, H0, W0)
        damp_edge_full = _resize_depth_like(final_damping[:, ii].reshape(B * E, 1, Hf, Wf), (Hp, Wp))
        damp_edge_full = damp_edge_full[..., :H0, :W0].view(B, E, 1, H0, W0)
        relative_pose = self._relative_from_c2w(final_poses, ii, jj)

        flow_mean = flow_full.abs().mean(dim=(2, 3, 4), keepdim=False).reshape(B * E, 1)
        conf_mean = conf_full.mean(dim=(2, 3, 4), keepdim=False).reshape(B * E, 1)
        depth_var = inv_edge_full.var(dim=(2, 3, 4), keepdim=False).reshape(B * E, 1)
        key_in = torch.cat(
            [
                edge_hidden.reshape(B * E, self.hidden_dim, Hf, Wf).mean(dim=(2, 3)),
                flow_mean,
                conf_mean,
                depth_var.clamp_max(10.0),
            ],
            dim=1,
        )
        keyframe_score = torch.sigmoid(self.keyframe_head(key_in)).view(B, E)

        return {
            "edges": list(edges),
            "edge_index_i": ii.detach().cpu(),
            "edge_index_j": jj.detach().cpu(),
            "spherical_flow": flow_full,
            "confidence": conf_full,
            "depth_confidence": conf_full,
            "inverse_depth": inv_edge_full,
            "damping": damp_edge_full,
            "relative_pose": relative_pose,
            "keyframe_score": keyframe_score,
            "poses_c2w_steps": poses_stack,
            "inverse_depth_steps": inv_stack,
            "residual_steps": residual_stack,
            "target_steps": target_stack,
            "weight_steps": weight_stack,
            "damping_steps": damping_stack,
            "refined_poses_c2w": final_poses,
            "refined_inverse_depth": final_inv,
            "initial_inverse_depth": inv_low_init,
            "flow_low": flow_low,
        }
