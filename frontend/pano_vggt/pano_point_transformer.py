"""Point-neighborhood transformer blocks for Pano-ReSplat Gaussian states."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


def _load_pointops(*, strict: bool) -> Any | None:
    try:
        import pointops  # type: ignore

        return pointops
    except Exception as exc:
        if strict:
            raise RuntimeError(
                "Refiner.knn_backend='pointops' requires the pointops extension. "
                "Install ReSplat pointops in the active environment before training."
            ) from exc
        return None


class PanoKNNTransformerBlock(nn.Module):
    """KNN self-attention block with a ReSplat-style pointops backend."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int = 4,
        knn: int = 8,
        mlp_ratio: float = 2.0,
        max_knn_points: int = 2048,
        chunk_size: int | None = None,
        attn_proj_channels: int | None = None,
        knn_backend: str = "cdist",
        strict_knn_backend: bool = False,
    ) -> None:
        super().__init__()
        if int(dim) <= 0:
            raise ValueError("dim must be positive.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.knn = max(1, int(knn))
        self.max_knn_points = int(max_knn_points)
        self.chunk_size = None if chunk_size is None else max(1, int(chunk_size))
        self.knn_backend = str(knn_backend).lower()
        if self.knn_backend not in {"cdist", "pointops"}:
            raise ValueError(f"Unsupported knn_backend: {knn_backend!r}")
        self.strict_knn_backend = bool(strict_knn_backend)
        self.attn_dim = int(attn_proj_channels) if attn_proj_channels is not None else self.dim
        if self.attn_dim <= 0:
            raise ValueError("attn_proj_channels must be positive when set.")
        if self.num_heads <= 0 or self.attn_dim % self.num_heads != 0:
            raise ValueError("num_heads must divide the attention channel count.")
        if self.knn_backend == "pointops" and self.num_heads != 1:
            raise ValueError("ReSplat-style pointops attention requires num_heads=1.")

        hidden = max(self.dim, int(round(float(mlp_ratio) * self.dim)))
        self.norm1 = nn.LayerNorm(self.dim)
        self.qkv = nn.Linear(self.dim, self.attn_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.attn_dim, self.dim)
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
        *,
        knn_cache: Any | None = None,
        return_knn_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        if xyz.ndim != 3 or int(xyz.shape[-1]) != 3:
            raise ValueError(f"xyz must have shape BxNx3, got {tuple(xyz.shape)}")
        if feat.ndim != 3 or int(feat.shape[-1]) != self.dim:
            raise ValueError(f"feat must have shape BxNx{self.dim}, got {tuple(feat.shape)}")
        if tuple(xyz.shape[:2]) != tuple(feat.shape[:2]):
            raise ValueError("xyz and feat must share B,N dimensions.")
        b, n, _ = [int(v) for v in xyz.shape]
        if n == 0:
            return (feat, knn_cache) if return_knn_cache else feat
        valid = None if valid_mask is None else valid_mask.to(device=xyz.device, dtype=torch.bool)
        if valid is not None and tuple(valid.shape) != (b, n):
            raise ValueError(f"valid_mask must have shape {(b, n)}, got {tuple(valid.shape)}")

        if self.knn_backend == "pointops":
            out, cache = self._forward_pointops(xyz, feat, valid, knn_cache)
        else:
            out, cache = self._forward_cdist(xyz, feat, valid, knn_cache)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return (out, cache) if return_knn_cache else out

    def _forward_cdist(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        valid: torch.Tensor | None,
        knn_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, _ = [int(v) for v in xyz.shape]
        idx = knn_cache if torch.is_tensor(knn_cache) else self._cdist_knn_indices(xyz, valid)
        x = self.norm1(feat)
        qkv = self.qkv(x)
        q, k_feat, v_feat = torch.chunk(qkv, chunks=3, dim=-1)
        q = q.view(b, n, self.num_heads, -1)
        k = self._gather_neighbors(k_feat, idx).view(b, n, idx.shape[-1], self.num_heads, -1)
        v_proj = self._gather_neighbors(v_feat, idx).view(b, n, idx.shape[-1], self.num_heads, -1)
        scores = (q.unsqueeze(2) * k).sum(dim=-1) / math.sqrt(float(q.shape[-1]))
        if valid is not None:
            neighbor_valid = self._gather_neighbors(valid.unsqueeze(-1).to(dtype=torch.float32), idx)[..., 0] > 0.5
            scores = scores.masked_fill(~neighbor_valid.unsqueeze(-1), -1.0e4)
        attn = torch.softmax(scores, dim=2)
        msg = (attn.unsqueeze(-1) * v_proj).sum(dim=2).reshape(b, n, self.attn_dim)
        if valid is not None:
            msg = msg * valid.unsqueeze(-1).to(dtype=msg.dtype)
        out = feat + self.out_proj(msg)
        out = out + self.mlp(self.norm2(out))
        if valid is not None:
            out = torch.where(valid.unsqueeze(-1), out, feat)
        return out, idx

    def _forward_pointops(
        self,
        xyz: torch.Tensor,
        feat: torch.Tensor,
        valid: torch.Tensor | None,
        knn_cache: list[torch.Tensor | None] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor | None]]:
        pointops = _load_pointops(strict=self.strict_knn_backend)
        if pointops is None:
            out, cache = self._forward_cdist(xyz, feat, valid, None)
            return out, [cache[i] for i in range(int(cache.shape[0]))]
        if not xyz.is_cuda:
            if self.strict_knn_backend:
                raise RuntimeError("pointops KNN is only supported for CUDA tensors in strict mode.")
            out, cache = self._forward_cdist(xyz, feat, valid, None)
            return out, [cache[i] for i in range(int(cache.shape[0]))]

        b, n, _ = [int(v) for v in xyz.shape]
        normalized = self.norm1(feat)
        out = feat.clone()
        caches: list[torch.Tensor | None] = []
        for batch_idx in range(b):
            valid_idx = (
                torch.arange(n, device=xyz.device)
                if valid is None
                else torch.nonzero(valid[batch_idx], as_tuple=False).flatten()
            )
            m = int(valid_idx.numel())
            if m == 0:
                caches.append(None)
                continue
            p = torch.nan_to_num(xyz[batch_idx].index_select(0, valid_idx).detach(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
            x = normalized[batch_idx].index_select(0, valid_idx)
            qkv = self.qkv(x)
            x_q, x_k, x_v = torch.chunk(qkv, chunks=3, dim=-1)
            offset = torch.tensor([m], device=xyz.device, dtype=torch.int32)
            k_count = min(self.knn, m)
            cached_idx = None
            if knn_cache is not None and batch_idx < len(knn_cache):
                cached_idx = knn_cache[batch_idx]
            if cached_idx is None:
                knn_idx, _ = pointops.knn_query(k_count, p, offset, p, offset)
            else:
                knn_idx = cached_idx
            grouped_k, knn_idx = pointops.knn_query_and_group(
                x_k.contiguous(),
                p,
                offset,
                new_xyz=p,
                new_offset=offset,
                idx=knn_idx,
                nsample=k_count,
                with_xyz=False,
            )
            grouped_v, _ = pointops.knn_query_and_group(
                x_v.contiguous(),
                p,
                offset,
                new_xyz=p,
                new_offset=offset,
                idx=knn_idx,
                nsample=k_count,
                with_xyz=False,
            )
            scores = (x_q.unsqueeze(1) * grouped_k).sum(dim=-1) / math.sqrt(float(self.attn_dim))
            msg = (torch.softmax(scores, dim=1).unsqueeze(-1) * grouped_v).sum(dim=1)
            updated = feat[batch_idx].index_select(0, valid_idx) + self.out_proj(msg)
            updated = updated + self.mlp(self.norm2(updated))
            out[batch_idx].index_copy_(0, valid_idx, updated)
            caches.append(knn_idx.detach())
        return out, caches

    def _cdist_knn_indices(self, xyz: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
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
