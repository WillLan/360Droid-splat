from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frontend.pano_vggt.dense_matcher import PoseGuidedDenseMatcher
from frontend.pano_vggt.engine import ExternalPanoVGGTInferenceEngine, build_panovggt_engine
from frontend.pano_vggt.matching_adapter import load_matching_sky_checkpoint
from frontend.pano_vggt.matching_head import PanoVGGTMatchingSkyHead
from frontend.pano_vggt.m3_config import parse_m3_sphere_config


def _eye_poses(n: int) -> torch.Tensor:
    return torch.eye(4).view(1, 4, 4).repeat(n, 1, 1)


def _depth(n: int = 2, hw: tuple[int, int] = (8, 12)) -> torch.Tensor:
    return torch.full((n, 1, hw[0], hw[1]), 2.0)


def _one_hot_descriptors(n: int = 2, feature_hw: tuple[int, int] = (4, 6)) -> torch.Tensor:
    h, w = feature_hw
    c = h * w
    desc = torch.zeros(n, c, h, w)
    for idx in range(c):
        y, x = divmod(idx, w)
        desc[:, idx, y, x] = 1.0
    return torch.nn.functional.normalize(desc, dim=1)


def _conf(n: int = 2, feature_hw: tuple[int, int] = (4, 6), value: float = 1.0) -> torch.Tensor:
    return torch.full((n, 1, feature_hw[0], feature_hw[1]), float(value))


def _run_matcher(
    desc: torch.Tensor,
    *,
    sky: torch.Tensor | None = None,
    radius: int = 1,
    fb: bool = True,
    fb_tolerance: float = 1.5,
    feature_hw: tuple[int, int] = (4, 6),
) -> object:
    matcher = PoseGuidedDenseMatcher(
        search_radius=radius,
        min_match_confidence=0.0,
        min_static_confidence=0.0,
        max_factors=4096,
        forward_backward=fb,
        fb_tolerance=fb_tolerance,
        depth_consistency_abs=0.1,
        depth_consistency_rel=0.1,
    )
    return matcher.match(
        poses_c2w=_eye_poses(2),
        depth=_depth(2, (8, 12)),
        dense_descriptors=desc,
        match_confidence=_conf(2, feature_hw),
        sky_prob=torch.zeros(2, 1, *feature_hw) if sky is None else sky,
        image_hw=(8, 12),
        feature_hw=feature_hw,
        edge_pairs=torch.tensor([[0, 1]]),
    )


def _write_bundle(path: Path, *, feature_dim: int = 8, descriptor_dim: int = 24, feature_hook: str | None = "hook") -> None:
    wrapper = PanoVGGTMatchingSkyHead(feature_dim, descriptor_dim=descriptor_dim, hidden_dim=8, num_conv_blocks=1)
    torch.save(
        {
            "format": "panovggt_m3_sphere_matching_sky_bundle_v1",
            "matching_head": wrapper.matching_head.state_dict(),
            "sky_mask_head": wrapper.sky_head.state_dict(),
            "descriptor_dim": descriptor_dim,
            "head_config": wrapper.head_config(),
            "feature_hook": feature_hook,
            "class_map": {"sky_ids": [1]},
        },
        path,
    )


def test_matching_head_disabled_keeps_fake_engine_old_path():
    engine = build_panovggt_engine(
        {
            "engine": "fake",
            "image_size": [8, 16],
            "M3Sphere": {"enabled": True},
            "MatchingHead": {"enabled": False},
        }
    )
    pred = engine.infer(torch.rand(2, 3, 8, 16))
    assert pred.dense_descriptors is None
    assert pred.match_confidence is None
    assert pred.sky_prob is None
    assert pred.matching_debug is None
    assert engine.last_dense_factor_graph is None


def test_fake_engine_matching_outputs_are_deterministic():
    cfg = {
        "engine": "fake",
        "image_size": [16, 32],
        "M3Sphere": {"enabled": True},
        "MatchingHead": {"enabled": True, "allow_fake_matching": True, "descriptor_dim": 24, "fake_feature_stride": 4},
        "DenseMatching": {"enabled": True, "max_samples_per_edge": 32, "search_radius": 1},
    }
    images = torch.rand(3, 3, 16, 32)
    first = build_panovggt_engine(cfg).infer(images)
    second = build_panovggt_engine(cfg).infer(images)
    assert torch.allclose(first.dense_descriptors, second.dense_descriptors)
    assert torch.allclose(first.match_confidence, second.match_confidence)
    assert torch.allclose(first.sky_prob, second.sky_prob)
    assert first.feature_hw == (4, 8)
    assert first.image_hw == (16, 32)
    assert first.matching_debug is not None


def test_tiny_checkpoint_loads_and_preserves_feature_hw(tmp_path: Path):
    path = tmp_path / "bundle.pt"
    _write_bundle(path, feature_dim=8, descriptor_dim=24)
    adapter = load_matching_sky_checkpoint(path, descriptor_dim=24, device="cpu")
    feature = torch.rand(1, 2, 8, 5, 7)
    out = adapter(feature)
    assert out["dense_descriptors"].shape == (2, 24, 5, 7)
    assert out["match_confidence"].shape == (2, 1, 5, 7)
    assert out["sky_prob"].shape == (2, 1, 5, 7)
    assert out["feature_hw"] == (5, 7)


def test_real_matching_without_hook_or_feature_method_raises_clear_error(tmp_path: Path):
    path = tmp_path / "bundle.pt"
    _write_bundle(path, feature_dim=8, descriptor_dim=24, feature_hook=None)
    adapter = load_matching_sky_checkpoint(path, descriptor_dim=24, device="cpu")

    class DummyModel(torch.nn.Module):
        def forward(self, images):
            return {"camera_poses": _eye_poses(images.shape[1]).unsqueeze(0), "depth": torch.ones(1, images.shape[1], 4, 8)}

    engine = object.__new__(ExternalPanoVGGTInferenceEngine)
    engine.model = DummyModel()
    engine.matching_adapter = adapter
    engine.patch_multiple = 1
    engine.m3_config = parse_m3_sphere_config({"PanoVGGT": {"M3Sphere": {"enabled": True}, "MatchingHead": {"enabled": True}}})
    with pytest.raises(RuntimeError, match="no feature_hook"):
        engine._call_model_and_maybe_capture_feature(torch.rand(1, 2, 3, 4, 8))


def test_identity_pose_identical_descriptors_match_to_self():
    feature_hw = (4, 6)
    graph = _run_matcher(_one_hot_descriptors(2, feature_hw), radius=0, feature_hw=feature_hw)
    factor = graph.factors[0]
    assert factor.valid_mask.all()
    assert torch.allclose(factor.tgt_uv, factor.src_uv, atol=1.0e-4)
    assert torch.allclose(factor.tgt_bearing, factor.src_bearing, atol=1.0e-5)


def test_horizontal_seam_wraparound_search_can_match_across_boundary():
    feature_hw = (2, 4)
    desc = torch.zeros(2, 4, *feature_hw)
    desc[0, 0, :, 3] = 1.0
    desc[1, 0, :, 0] = 1.0
    desc[:, 1:, :, :] = 0.01
    desc = torch.nn.functional.normalize(desc, dim=1)
    graph = _run_matcher(desc, radius=1, fb=False, feature_hw=feature_hw)
    factor = graph.factors[0]
    seam = factor.src_uv[:, 0] > 3.0
    assert seam.any()
    assert torch.all(factor.tgt_uv[seam, 0] < 1.0)


def test_local_search_recovers_known_shifted_descriptor_peak():
    feature_hw = (3, 5)
    src = _one_hot_descriptors(1, feature_hw)[0]
    tgt = torch.roll(src, shifts=1, dims=-1)
    desc = torch.stack([src, tgt], dim=0)
    graph = _run_matcher(desc, radius=1, feature_hw=feature_hw)
    factor = graph.factors[0]
    interior = (factor.src_uv[:, 0] > 1.0) & (factor.src_uv[:, 0] < 4.0)
    assert torch.allclose(factor.tgt_uv[interior, 0], factor.src_uv[interior, 0] + 1.0, atol=1.0e-4)


def test_forward_backward_check_filters_inconsistent_matches():
    feature_hw = (2, 5)
    desc = torch.zeros(2, 3, *feature_hw)
    desc[0, 0, :, 1] = 1.0
    desc[0, 0, :, 3] = 1.0
    desc[1, 0, :, 2] = 1.0
    desc[:, 1, :, :] = 0.1
    desc[:, 2, :, :] = 0.2
    desc = torch.nn.functional.normalize(desc, dim=1)
    graph = _run_matcher(desc, radius=3, fb=True, fb_tolerance=0.1, feature_hw=feature_hw)
    factor = graph.factors[0]
    assert factor.metadata["fb_pass_mask"].logical_not().any()
    assert factor.valid_mask.logical_not().any()


def test_high_sky_probability_reduces_factor_weight():
    feature_hw = (4, 6)
    desc = _one_hot_descriptors(2, feature_hw)
    clear = _run_matcher(desc, sky=torch.zeros(2, 1, *feature_hw), radius=0, feature_hw=feature_hw)
    sky = torch.zeros(2, 1, *feature_hw)
    sky[:, :, :, :] = 0.95
    cloudy = _run_matcher(desc, sky=sky, radius=0, feature_hw=feature_hw)
    assert cloudy.factors[0].weight.mean() < clear.factors[0].weight.mean()


def test_factor_weights_are_finite_and_clamped():
    feature_hw = (4, 6)
    graph = _run_matcher(_one_hot_descriptors(2, feature_hw), radius=0, feature_hw=feature_hw)
    weight = graph.factors[0].weight
    assert torch.isfinite(weight).all()
    assert weight.min() >= 0.0
    assert weight.max() <= 1.0
