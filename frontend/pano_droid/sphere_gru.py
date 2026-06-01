"""SphereConvGRU update block."""

from __future__ import annotations

import warnings

import torch
from torch import nn

from .sphere_conv import SphereConv2d


class SphereConvGRU(nn.Module):
    """ConvGRU with 3x3/5x5 convolutions replaced by ``SphereConv2d``."""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size not in (1, 3, 5):
            raise ValueError("SphereConvGRU supports kernel_size 1, 3, or 5.")
        conv_cls = nn.Conv2d if kernel_size == 1 else SphereConv2d
        gate_dim = input_dim + hidden_dim
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = int(kernel_size)
        self.convz = conv_cls(gate_dim, hidden_dim, kernel_size)
        self.convr = conv_cls(gate_dim, hidden_dim, kernel_size)
        self.convq = conv_cls(gate_dim, hidden_dim, kernel_size)

    def forward(self, h: torch.Tensor | None, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x as BxCxHxW, got {tuple(x.shape)}")
        if h is None:
            h = x.new_zeros(x.shape[0], self.hidden_dim, x.shape[-2], x.shape[-1])
        if h.shape[0] != x.shape[0] or h.shape[-2:] != x.shape[-2:]:
            raise ValueError(
                f"Hidden shape {tuple(h.shape)} is incompatible with input {tuple(x.shape)}"
            )
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1.0 - z) * h + z * q

    def load_from_convgru_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        prefix: str = "",
        strict_shapes: bool = False,
    ) -> list[str]:
        """Best-effort initialization from a regular ConvGRU state dict.

        Returns the list of copied parameter names and warns for incompatible
        shapes instead of silently skipping them.
        """
        own = self.state_dict()
        copied = []
        incompatible = []
        for own_name, own_tensor in own.items():
            candidates = [prefix + own_name, own_name]
            if ".conv." in own_name:
                candidates.append(prefix + own_name.replace(".conv.", "."))
                candidates.append(own_name.replace(".conv.", "."))
            src_name = next((name for name in candidates if name in state_dict), None)
            if src_name is None:
                continue
            src = state_dict[src_name]
            if tuple(src.shape) != tuple(own_tensor.shape):
                incompatible.append((own_name, tuple(src.shape), tuple(own_tensor.shape)))
                if strict_shapes:
                    raise ValueError(
                        f"Shape mismatch for {own_name}: source {tuple(src.shape)} "
                        f"vs target {tuple(own_tensor.shape)}"
                    )
                continue
            own_tensor.copy_(src)
            copied.append(own_name)
        self.load_state_dict(own)
        if incompatible:
            msg = "; ".join(f"{n}: {s}->{t}" for n, s, t in incompatible[:8])
            warnings.warn(
                "Some ConvGRU weights were not loaded into SphereConvGRU due to "
                f"shape mismatch: {msg}",
                RuntimeWarning,
                stacklevel=2,
            )
        if not copied:
            warnings.warn(
                "No compatible ConvGRU weights were found for SphereConvGRU.",
                RuntimeWarning,
                stacklevel=2,
            )
        return copied

