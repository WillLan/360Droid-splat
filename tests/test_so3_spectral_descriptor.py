from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from frontend.pano_droid.spherical_ba import se3_exp
from frontend.spherical_selfi.panorama_loop import PanoramaLoopDetector
from geometry.spherical_erp import build_erp_ray_grid
from geometry.spherical_spectral_descriptor import build_so3_sh_gram_descriptor


def _analytic_adapter_field(direction: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    x, y, z = direction.unbind(dim=-1)
    longitude = torch.atan2(x, z)
    signal = torch.stack(
        [
            x,
            y,
            z,
            x * y,
            y * z,
            z * x,
            x.square() - y.square(),
            3.0 * z.square() - 1.0,
            torch.sin(2.0 * longitude) * (1.0 - y.square()),
            torch.cos(3.0 * longitude) * (1.0 - y.square()),
        ],
        dim=-1,
    )
    return torch.einsum("...k,ck->c...", signal, projection)


def test_so3_sh_gram_descriptor_has_expected_dimension_and_arbitrary_rotation_invariance() -> None:
    torch.manual_seed(7)
    height, width = 64, 128
    rays = build_erp_ray_grid(height, width)
    projection = torch.randn(24, 10)
    rotation = se3_exp(
        torch.tensor([0.0, 0.0, 0.0, 0.70, -0.40, 0.30])
    )[:3, :3]
    original = _analytic_adapter_field(rays, projection).unsqueeze(0)
    rotated = _analytic_adapter_field(rays @ rotation.T, projection).unsqueeze(0)

    first = build_so3_sh_gram_descriptor(original, max_degree=6, num_samples=4096)
    second = build_so3_sh_gram_descriptor(rotated, max_degree=6, num_samples=4096)

    assert first.shape == second.shape == (1, 2107)
    assert float(F.cosine_similarity(first, second)) > 0.9999
    assert float((first - second).norm()) < 0.01


def test_so3_window_retrieval_uses_best_frame_pair_and_candidate_nms() -> None:
    detector = PanoramaLoopDetector(
        descriptor_mode="so3_sh_gram",
        top_k=20,
        exclude_recent_windows=0,
        min_retrieval_score=0.5,
        candidate_nms_radius=2,
    )
    query = F.normalize(torch.eye(4, 8), dim=-1)

    def packet(window_id: int, descriptors: torch.Tensor):
        return SimpleNamespace(
            window_id=window_id,
            retrieval_descriptors=F.normalize(descriptors, dim=-1),
        )

    close = torch.randn(4, 8)
    close[2] = query[3]
    close_neighbor = torch.randn(4, 8)
    close_neighbor[1] = 0.99 * query[3] + 0.01 * query[2]
    distant = torch.randn(4, 8)
    distant[0] = query[1]
    detector.add(packet(0, close))
    detector.add(packet(1, close_neighbor))
    detector.add(packet(5, distant))

    candidates = detector.retrieve(packet(10, query))

    assert [item.packet.window_id for item in candidates] == [0, 5]
    assert candidates[0].source_frame_index == 2
    assert candidates[0].target_frame_index == 3
    assert candidates[1].source_frame_index == 0
    assert candidates[1].target_frame_index == 1
    assert detector._descriptor_database is not None
    assert detector._descriptor_database.dtype == torch.float16
    assert detector.descriptor_database_bytes == 3 * 4 * 8 * 2
