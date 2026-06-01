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


def _resize_like(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bilinear", align_corners=True)


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
            nn.Conv2d(self.context_dim + corr_dim + 2, self.hidden_dim, 3, padding=1),
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
        # The implementation is local-on-demand, so ``max_corr_elements`` is a
        # compatibility guard for future all-pairs variants rather than a memory
        # allocation trigger.
        return SphericalCorrBlock(
            fmap0,
            fmap1,
            num_levels=self.corr_levels,
            radius=self.corr_radius,
            latitude_scale=self.use_spherical_corr,
        )

    def forward(
        self,
        image0: torch.Tensor,
        image1: Optional[torch.Tensor] = None,
        *,
        num_updates: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
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
            gru_in = torch.cat([context, corr, flow], dim=1)
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
    ) -> dict[str, torch.Tensor | list[tuple[int, int]]]:
        """Run pairwise DROID-style updates for a multi-frame graph.

        ``images`` is ``B x N x C x H x W`` and ``edges`` stores source/target
        frame indices.  The returned tensors are grouped as ``B x E ...``.
        """
        if images.ndim != 5:
            raise ValueError(f"Expected images as BxNxCxHxW, got {tuple(images.shape)}")
        B, N, C, H, W = images.shape
        if not edges:
            raise ValueError("forward_graph requires at least one edge.")
        flows = []
        confs = []
        inv_depths = []
        dampings = []
        poses = []
        key_scores = []
        for i, j in edges:
            if i < 0 or i >= N or j < 0 or j >= N:
                raise IndexError(f"Graph edge {(i, j)} is outside sequence length {N}.")
            pred = self(
                images[:, i],
                images[:, j],
                num_updates=num_updates,
            )
            flows.append(pred["spherical_flow"])
            confs.append(pred["confidence"])
            inv_depths.append(pred["inverse_depth"])
            dampings.append(pred["damping"])
            poses.append(pred["relative_pose"])
            key_scores.append(pred["keyframe_score"])
        return {
            "edges": list(edges),
            "spherical_flow": torch.stack(flows, dim=1),
            "confidence": torch.stack(confs, dim=1),
            "depth_confidence": torch.stack(confs, dim=1),
            "inverse_depth": torch.stack(inv_depths, dim=1),
            "damping": torch.stack(dampings, dim=1),
            "relative_pose": torch.stack(poses, dim=1),
            "keyframe_score": torch.stack(key_scores, dim=1),
        }
