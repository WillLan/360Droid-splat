"""Losses and graph helpers for DROID-style PanoDROID training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .projective_ops import projective_flow_from_depth
from .spherical_camera import latitude_area_weight, pixel_grid


def build_temporal_edges(n_frames: int, radius: int = 2, *, bidirectional: bool = True) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for i in range(int(n_frames)):
        for d in range(1, int(radius) + 1):
            j = i + d
            if j >= int(n_frames):
                continue
            edges.append((i, j))
            if bidirectional:
                edges.append((j, i))
    return edges


def build_proximity_edges(
    poses_c2w: torch.Tensor,
    depths: torch.Tensor | None = None,
    *,
    radius: int = 2,
    max_edges: int = 0,
    bidirectional: bool = True,
    sample_height: int = 8,
    sample_width: int = 16,
) -> list[tuple[int, int]]:
    """Build projection-distance/co-visibility graph edges."""
    if poses_c2w.ndim == 4:
        poses = poses_c2w[0]
    else:
        poses = poses_c2w
    depth_maps = None
    if depths is not None:
        depth_maps = depths[0] if depths.ndim == 5 else depths
    n_frames = int(poses.shape[0])
    temporal = build_temporal_edges(n_frames, radius=radius, bidirectional=bidirectional)
    edge_set = set(temporal)
    candidates: list[tuple[float, int, int]] = []
    if depth_maps is not None:
        H, W = int(depth_maps.shape[-2]), int(depth_maps.shape[-1])
        sy = max(1, H // max(1, int(sample_height)))
        sx = max(1, W // max(1, int(sample_width)))
        pixels = pixel_grid(H, W, device=poses.device, dtype=poses.dtype)[::sy, ::sx].reshape(-1, 2)
    else:
        centers = poses[:, :3, 3].detach().float()
        dist = torch.cdist(centers, centers)
    for i in range(n_frames):
        for j in range(n_frames):
            if i == j or (i, j) in edge_set:
                continue
            if depth_maps is None:
                score = float(dist[i, j].cpu())
            else:
                depth = depth_maps[i, 0, ::sy, ::sx].reshape(1, -1).to(poses)
                valid = depth > 1e-6
                if not bool(valid.any()):
                    score = float("inf")
                else:
                    flow, _ = _projective_target(
                        pixels,
                        depth,
                        poses[i : i + 1],
                        poses[j : j + 1],
                        height=H,
                        width=W,
                    )
                    score = float(torch.sqrt((flow * flow).sum(dim=-1) + 1e-6)[valid].mean().detach().cpu())
            candidates.append((score, i, j))
    candidates.sort(key=lambda x: x[0])
    target = int(max_edges) if int(max_edges) > 0 else max(len(temporal), n_frames * max(1, int(radius)) * 2)
    out = list(temporal)
    for _, i, j in candidates:
        if len(out) >= target:
            break
        out.append((i, j))
        if bidirectional and len(out) < target:
            out.append((j, i))
    return out


def select_training_edges(
    edges: list[tuple[int, int]],
    *,
    max_edges: int = 0,
    n_frames: int | None = None,
    generator: torch.Generator | None = None,
) -> list[tuple[int, int]]:
    """Randomly sample graph edges while trying to keep frame coverage."""
    if not edges:
        return []
    limit = int(max_edges)
    if limit <= 0 or limit >= len(edges):
        return list(edges)
    perm = torch.randperm(len(edges), generator=generator).tolist()
    chosen: list[tuple[int, int]] = []
    covered: set[int] = set()
    if n_frames is not None:
        for idx in perm:
            edge = edges[idx]
            adds_coverage = edge[0] not in covered or edge[1] not in covered
            if adds_coverage:
                chosen.append(edge)
                covered.update(edge)
            if len(chosen) >= limit or len(covered) >= int(n_frames):
                break
    for idx in perm:
        if len(chosen) >= limit:
            break
        edge = edges[idx]
        if edge not in chosen:
            chosen.append(edge)
    return chosen


@dataclass
class GraphLossWeights:
    pose: float = 10.0
    flow: float = 0.05
    residual: float = 0.01
    depth: float = 0.0
    smooth: float = 0.0
    confidence: float = 0.0


def _relative_pose_from_c2w(c2w_i: torch.Tensor, c2w_j: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(c2w_j) @ c2w_i


def _pose_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    R = pred[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2)
    trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    rot = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
    trans = (pred[:, :3, 3] - target[:, :3, 3]).abs().mean(dim=-1)
    return (rot + trans).mean()


def _smoothness(x: torch.Tensor) -> torch.Tensor:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx_wrap = x[..., :, :1] - x[..., :, -1:]
    return (
        torch.sqrt(dx * dx + 1e-6).mean()
        + torch.sqrt(dy * dy + 1e-6).mean()
        + torch.sqrt(dx_wrap * dx_wrap + 1e-6).mean()
    )


def _downsample_depths(depths: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    B, N = depths.shape[:2]
    low = F.interpolate(depths.reshape(B * N, 1, depths.shape[-2], depths.shape[-1]), size=size, mode="nearest")
    return low.view(B, N, 1, size[0], size[1])


def _projective_target(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    c2w_i: torch.Tensor,
    c2w_j: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return projective_flow_from_depth(
        pixels,
        depth,
        c2w_i,
        c2w_j,
        height=height,
        width=width,
    )


def spherical_projective_flow(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    c2w_i: torch.Tensor,
    c2w_j: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source pixels/depth from camera i into camera j."""
    return _projective_target(pixels, depth, c2w_i, c2w_j, height=height, width=width)


def graph_supervised_loss(
    batch: dict,
    pred: dict,
    *,
    weights: GraphLossWeights | None = None,
    sample_height: int | None = None,
    sample_width: int | None = None,
    gamma: float = 0.9,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or GraphLossWeights()
    device = pred["refined_poses_c2w"].device
    dtype = pred["refined_poses_c2w"].dtype
    depths = batch["depths"].to(device=device, dtype=dtype)
    gt_poses = batch["poses_c2w"].to(device=device, dtype=dtype)
    edges = list(pred["edges"])
    B, N = gt_poses.shape[:2]
    inv_steps = pred["inverse_depth_steps"]
    pose_steps = pred["poses_c2w_steps"]
    residual_steps = pred["residual_steps"]
    weight_steps = pred["weight_steps"]
    _, S, _, _, Hf, Wf = inv_steps.shape
    pixels = pixel_grid(Hf, Wf, device=device, dtype=dtype).reshape(-1, 2)
    if sample_height is not None and sample_width is not None:
        # Backward-compatible knobs now select a deterministic subset of the feature grid.
        sy = max(1, Hf // max(1, int(sample_height)))
        sx = max(1, Wf // max(1, int(sample_width)))
    else:
        sy = sx = 1
    keep = pixel_grid(Hf, Wf, device=device, dtype=dtype)[::sy, ::sx].reshape(-1, 2)
    pixels = keep
    low_depth = _downsample_depths(depths, (Hf, Wf))
    low_inv_gt = torch.zeros_like(low_depth)
    valid_depth = low_depth > 1e-6
    low_inv_gt[valid_depth] = 1.0 / low_depth[valid_depth].clamp_min(1e-6)
    area = latitude_area_weight(Hf, Wf, device=device, dtype=dtype, normalize=True)[0, ::sy, ::sx].reshape(1, -1)

    l_pose = depths.new_tensor(0.0)
    l_flow = depths.new_tensor(0.0)
    l_depth = depths.new_tensor(0.0)
    l_residual = depths.new_tensor(0.0)
    l_smooth = depths.new_tensor(0.0)
    l_conf = depths.new_tensor(0.0)
    valid_edges = 0

    for s in range(S):
        step_w = float(gamma) ** (S - s - 1)
        poses_s = pose_steps[:, s]
        inv_s = inv_steps[:, s]
        inv_sample = inv_s[..., ::sy, ::sx].reshape(B, N, -1)
        gt_inv_sample = low_inv_gt[..., ::sy, ::sx].reshape(B, N, -1)
        valid_inv_sample = valid_depth[..., ::sy, ::sx].reshape(B, N, -1)
        depth_weight = area.unsqueeze(1) * valid_inv_sample.to(dtype)
        l_depth = l_depth + step_w * (
            torch.sqrt((inv_sample - gt_inv_sample) ** 2 + 1e-6) * depth_weight
        ).sum() / depth_weight.sum().clamp_min(1e-6)
        l_smooth = l_smooth + step_w * _smoothness(inv_s)

        for e, (i, j) in enumerate(edges):
            gt_depth = low_depth[:, i, 0, ::sy, ::sx].reshape(B, -1)
            valid = gt_depth > 1e-6
            if not bool(valid.any()):
                continue
            gt_rel = _relative_pose_from_c2w(gt_poses[:, i], gt_poses[:, j])
            pred_rel = _relative_pose_from_c2w(poses_s[:, i], poses_s[:, j])
            l_pose = l_pose + step_w * _pose_loss(pred_rel, gt_rel)

            gt_flow, _ = _projective_target(
                pixels,
                gt_depth,
                gt_poses[:, i],
                gt_poses[:, j],
                height=Hf,
                width=Wf,
            )
            pred_depth = torch.zeros_like(inv_sample[:, i])
            pred_valid = inv_sample[:, i] > 1e-6
            pred_depth[pred_valid] = 1.0 / inv_sample[:, i][pred_valid].clamp_min(1e-6)
            pred_flow, _ = _projective_target(
                pixels,
                pred_depth,
                poses_s[:, i],
                poses_s[:, j],
                height=Hf,
                width=Wf,
            )
            flow_err = pred_flow - gt_flow
            edge_area = area * valid.to(dtype)
            l_flow = l_flow + step_w * (
                torch.sqrt((flow_err * flow_err).sum(dim=-1) + 1e-6) * edge_area
            ).sum() / edge_area.sum().clamp_min(1e-6)

            res = residual_steps[:, s, e, ::sy, ::sx].reshape(B, -1, 2)
            weight_e = weight_steps[:, s, e, :, ::sy, ::sx].reshape(B, weight_steps.shape[3], -1)
            conf = weight_e.mean(dim=1).clamp(1e-4, 1.0)
            res_norm = torch.sqrt((res * res).sum(dim=-1) + 1e-6)
            l_residual = l_residual + step_w * (
                res_norm * conf * edge_area
            ).sum() / edge_area.sum().clamp_min(1e-6)
            l_conf = l_conf + step_w * (
                conf * res_norm.detach() - torch.log(conf)
            ).mean()
            valid_edges += 1

    denom = max(valid_edges, 1)
    l_pose = l_pose / denom
    l_flow = l_flow / denom
    l_residual = l_residual / denom
    l_conf = l_conf / max(S, 1)
    l_depth = l_depth / max(S, 1)
    l_smooth = l_smooth / max(S, 1)
    total = (
        weights.pose * l_pose
        + weights.flow * l_flow
        + weights.depth * l_depth
        + weights.residual * l_residual
        + weights.smooth * l_smooth
        + weights.confidence * l_conf
    )
    mean_residual = residual_steps.detach().norm(dim=-1).mean()
    damping = pred["damping_steps"].detach()
    covered = len({idx for edge in edges for idx in edge})
    valid_mask = pred.get("ba_valid_mask")
    valid_ratio = (
        valid_mask.detach().float().mean()
        if torch.is_tensor(valid_mask)
        else torch.tensor(1.0, device=device)
    )
    upmask = pred.get("upmask_steps")
    if torch.is_tensor(upmask):
        up = upmask.detach().reshape(-1, 9, 8, 8, upmask.shape[-2], upmask.shape[-1])
        prob = torch.softmax(up, dim=1).clamp_min(1e-8)
        up_entropy = -(prob * prob.log()).sum(dim=1).mean()
    else:
        up_entropy = torch.tensor(0.0, device=device)
    pose_update_norm = pred.get("ba_pose_update_norm")
    depth_update_norm = pred.get("ba_depth_update_norm")
    normal_condition = pred.get("ba_normal_condition")
    return total, {
        "loss": total.detach(),
        "pose": l_pose.detach(),
        "flow": l_flow.detach(),
        "depth": l_depth.detach(),
        "residual": l_residual.detach(),
        "smooth": l_smooth.detach(),
        "confidence": l_conf.detach(),
        "ba_residual": mean_residual,
        "ba_valid_ratio": valid_ratio,
        "ba_pose_update_norm": pose_update_norm.detach() if torch.is_tensor(pose_update_norm) else torch.tensor(0.0, device=device),
        "ba_depth_update_norm": depth_update_norm.detach() if torch.is_tensor(depth_update_norm) else torch.tensor(0.0, device=device),
        "ba_normal_condition": normal_condition.detach() if torch.is_tensor(normal_condition) else torch.tensor(0.0, device=device),
        "damping_mean": damping.mean(),
        "damping_max": damping.max(),
        "upmask_entropy": up_entropy,
        "valid_edges": torch.tensor(float(valid_edges), device=device),
        "edge_coverage": torch.tensor(float(covered) / max(float(N), 1.0), device=device),
    }
