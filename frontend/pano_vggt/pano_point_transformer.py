"""Point-neighborhood transformer blocks for Pano-ReSplat Gaussian states."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class PanoKNNTransformerBlock(nn.Module):
    """A small KNN self-attention block over Gaussian centers."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        knn: int = 8,
        mlp_ratio: float = 2.0,
        max_knn_points: int = 2048,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive.")
        if int(num_heads) <= 0 or int(dim) % int(num_heads) != 0:
            raise ValueError("num_heads must divide dim.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.knn = max(1, int(knn))
        self.max_knn_points = int(max_knn_points)
        self.chunk_size = None if chunk_size is None else max(1, int(chunk_size))
        hidden = max(self.dim, int(round(float(mlp_ratio) * self.dim)))

        self.norm1 = nn.LayerNorm(self.dim)
        self.q_proj = nn.Linear(self.dim, self.dim)
        self.k_proj = nn.Linear(self.dim, self.dim)
        self.v_proj = nn.Linear(self.dim, self.dim)
        self.out_proj = nn.Linear(self.dim, self.dim)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.dim),
        )

    def forward(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if xyz.ndim != 3 or int(xyz.shape[-1]) != 3:
            raise ValueError(f"xyz must have shape BxNx3, got {tuple(xyz.shape)}")
        if feat.ndim != 3 or int(feat.shape[-1]) != self.dim:
            raise ValueError(f"feat must have shape BxNx{self.dim}, got {tuple(feat.shape)}")
        if tuple(xyz.shape[:2]) != tuple(feat.shape[:2]):
            raise ValueError("xyz and feat must share B,N dimensions.")
        b, n, _ = [int(v) for v in xyz.shape]
        if n == 0:
            return feat
        valid = None if valid_mask is None else valid_mask.to(device=xyz.device, dtype=torch.bool)
        if valid is not None and tuple(valid.shape) != (b, n):
            raise ValueError(f"valid_mask must have shape {(b, n)}, got {tuple(valid.shape)}")

        idx = self._knn_indices(xyz, valid)
        x = self.norm1(feat)
        q = self.q_proj(x).view(b, n, self.num_heads, -1)
        k = self._gather_neighbors(self.k_proj(x), idx).view(b, n, idx.shape[-1], self.num_heads, -1)
        v_proj = self._gather_neighbors(self.v_proj(x), idx).view(b, n, idx.shape[-1], self.num_heads, -1)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(float(q.shape[-1]))
        if valid is not None:
            neighbor_valid = self._gather_neighbors(valid.unsqueeze(-1).to(dtype=torch.float32), idx)[..., 0] > 0.5
            scores = scores.masked_fill(~neighbor_valid.unsqueeze(-1), -1.0e4)
        attn = torch.softmax(scores, dim=2)
        msg = (attn.unsqueeze(-1) * v_proj).sum(dim=2).reshape(b, n, self.dim)
        if valid is not None:
            msg = msg * valid.unsqueeze(-1).to(dtype=msg.dtype)
        out = feat + self.out_proj(msg)
        out = out + self.mlp(self.norm2(out))
        if valid is not None:
            out = torch.where(valid.unsqueeze(-1), out, feat)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def _knn_indices(self, xyz: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        b, n, _ = [int(v) for v in xyz.shape]
        k = min(self.knn, n)
        safe_xyz = torch.nan_to_num(xyz.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        ref_indices = None
        search_xyz = safe_xyz
        search_valid = valid
        if self.max_knn_points > 0 and n > self.max_knn_points:
            ref_indices = torch.linspace(0, n - 1, steps=int(self.max_knn_points), device=xyz.device).round().long()
            search_xyz = safe_xyz.index_select(1, ref_indices)
            search_valid = None if valid is None else valid.index_select(1, ref_indices)
            k = min(k, int(search_xyz.shape[1]))
        chunk = self.chunk_size
        if chunk is None and n > 4096:
            chunk = 4096
        if chunk is None or chunk >= n:
            dist = torch.cdist(safe_xyz, search_xyz)
            if search_valid is not None:
                dist = dist.masked_fill(~search_valid[:, None, :], float("inf"))
            idx = torch.topk(dist, k=k, dim=-1, largest=False).indices
            return ref_indices[idx] if ref_indices is not None else idx

        parts = []
        for start in range(0, n, int(chunk)):
            stop = min(start + int(chunk), n)
            dist = torch.cdist(safe_xyz[:, start:stop], search_xyz)
            if search_valid is not None:
                dist = dist.masked_fill(~search_valid[:, None, :], float("inf"))
            idx = torch.topk(dist, k=k, dim=-1, largest=False).indices
            parts.append(ref_indices[idx] if ref_indices is not None else idx)
        return torch.cat(parts, dim=1)

    @staticmethod
    def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        gathered = []
        b, n, k = [int(v) for v in indices.shape]
        for batch_idx in range(b):
            flat = indices[batch_idx].reshape(-1)
            gathered.append(values[batch_idx].index_select(0, flat).view(n, k, *values.shape[2:]))
        return torch.stack(gathered, dim=0)
