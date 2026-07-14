from __future__ import annotations

import torch

from models.sphereglue_local_ba import (
    SphereGlueLocalBAMatcher,
    pure_torch_knn_graph,
    sphereglue_unit_cartesian,
)


class _DummySuperPoint(torch.nn.Module):
    def extract(self, image: torch.Tensor, *, resize=None):
        assert resize is None
        keypoints = image.new_tensor(
            [[1.0, 1.0], [3.0, 2.0], [5.0, 3.0], [7.0, 4.0], [9.0, 5.0]]
        )
        descriptors = image.new_zeros(5, 256)
        descriptors[:, :5] = torch.eye(5, device=image.device, dtype=image.dtype)
        scores = image.new_tensor([0.99, 0.9, 0.8, 0.7, 0.6])
        return {
            "keypoints": keypoints.unsqueeze(0),
            "descriptors": descriptors.unsqueeze(0),
            "keypoint_scores": scores.unsqueeze(0),
        }


class _DummySphereGlue(torch.nn.Module):
    def forward(self, data):
        first = int(data["h1"].shape[1])
        second = int(data["h2"].shape[1])
        matches = torch.full(
            (1, first), -1, device=data["h1"].device, dtype=torch.long
        )
        count = min(first, second)
        matches[0, :count] = torch.arange(count, device=matches.device)
        confidence = torch.zeros(1, first, device=matches.device)
        confidence[0, :count] = 0.8
        return {"matches0": matches, "matching_scores0": confidence}


def test_pure_torch_knn_graph_is_deterministic_and_excludes_self() -> None:
    points = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )
    first = pure_torch_knn_graph(points, 2, flow="target_to_source")
    second = pure_torch_knn_graph(points, 2, flow="target_to_source")
    assert torch.equal(first, second)
    assert first.shape == (2, 8)
    assert not torch.any(first[0] == first[1])


def test_sphereglue_coordinate_conversion_matches_poles_and_unit_norm() -> None:
    keypoints = torch.tensor([[7.5, -0.5], [7.5, 7.5], [-0.5, 3.5]])
    unit = sphereglue_unit_cartesian(keypoints, height=8, width=16)
    torch.testing.assert_close(unit.norm(dim=-1), torch.ones(3), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(unit[0], torch.tensor([0.0, 0.0, 1.0]), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(unit[1], torch.tensor([0.0, 0.0, -1.0]), atol=1.0e-6, rtol=1.0e-6)


def test_sphereglue_cache_uses_sparse_mutual_matches_and_static_mask() -> None:
    images = torch.rand(1, 3, 3, 8, 16)
    depth = torch.full((1, 3, 1, 8, 16), 2.0)
    static = torch.ones_like(depth, dtype=torch.bool)
    static[:, :, :, 1, 1] = False
    matcher = SphereGlueLocalBAMatcher(
        {
            "num_queries": 4,
            "extractor_max_keypoints": 5,
            "knn": 2,
            "min_factor_weight": 0.1,
        },
        device="cpu",
        superpoint=_DummySuperPoint(),
        sphereglue=_DummySphereGlue(),
        provenance={"test_double": True},
    )
    cache = matcher.build_cache(images, depth, static_valid_mask=static)
    assert cache.source_uv.shape == (1, 3, 4, 2)
    assert cache.target_uv.shape == (1, 6, 4, 2)
    assert cache.num_factors == 24
    assert cache.source_valid.all()
    assert cache.valid_mask.all()
    assert cache.mutual_mask is not None and cache.mutual_mask.all()
    assert cache.metadata["matcher"] == "superpoint_sphereglue"
    assert cache.metadata["per_view_keypoints"] == [4, 4, 4]
    assert cache.metadata["source_area_reweight"] is False
    torch.testing.assert_close(
        cache.source_ray.norm(dim=-1), torch.ones_like(cache.source_depth)
    )

