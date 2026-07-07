from __future__ import annotations

import torch

from backend.pano_gs.mapper import PanoGaussianMap, PanoGaussianMapper
from frontend.pano_vggt.resplat_types import PanoGaussianState


def _state(
    means: torch.Tensor,
    *,
    log_scales: torch.Tensor | None = None,
    opacity_logits: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> PanoGaussianState:
    n = int(means.shape[0])
    log_s = torch.full((n, 3), -3.0) if log_scales is None else log_scales
    opacity = torch.full((n, 1), 2.0) if opacity_logits is None else opacity_logits
    rotations = torch.zeros(n, 4)
    rotations[:, 0] = 1.0
    sh = torch.zeros(n, 3, 9)
    sh[:, :, 1:] = 0.01
    latent = torch.zeros(n, 4)
    return PanoGaussianState(
        means=means.view(1, n, 3),
        log_scales=log_s.view(1, n, 3),
        rotations_unnorm=rotations.view(1, n, 4),
        opacity_logits=opacity.view(1, n, 1),
        sh_coeffs=sh.view(1, n, 3, 9),
        latent_features=latent.view(1, n, 4),
        source_view_ids=torch.zeros(1, n, dtype=torch.long),
        source_uv=torch.zeros(1, n, 2),
        valid_mask=torch.ones(1, n, dtype=torch.bool),
        confidence=(torch.ones(n, 1) if confidence is None else confidence).view(1, n, 1),
    )


def test_resplat_state_fusion_insert_preserves_parameters() -> None:
    state = _state(
        torch.tensor([[0.0, 0.0, 1.0], [0.4, 0.0, 1.0]]),
        log_scales=torch.log(torch.tensor([[0.03, 0.04, 0.05], [0.02, 0.03, 0.04]])),
        opacity_logits=torch.tensor([[1.0], [2.0]]),
    )
    gaussian_map = PanoGaussianMap(config={"BackendOptimization": {"sh_degree": 2}}, device="cpu")

    stats = gaussian_map.add_or_fuse_resplat_gaussians(
        state,
        frame_ids=[0, 1, 2, 3],
        config={"voxel_size": 0.1, "merge_radius": 0.05, "min_confidence": 0.0, "min_opacity": 0.0},
    )

    assert stats["inserted"] == 2
    assert gaussian_map.anchor_count() == 2
    order = torch.argsort(gaussian_map.get_xyz.detach()[:, 0])
    assert torch.allclose(gaussian_map.get_xyz.detach()[order], state.means[0], atol=1.0e-6)
    assert torch.allclose(gaussian_map.get_scaling.detach()[order], torch.exp(state.log_scales[0]), atol=2.0e-5)
    assert torch.allclose(gaussian_map.get_rotation.detach()[order], state.rotations_unnorm[0], atol=1.0e-6)
    assert torch.allclose(gaussian_map.get_opacity.detach()[order], torch.sigmoid(state.opacity_logits[0]), atol=1.0e-5)
    assert gaussian_map.get_sh_coefficients.shape == (2, 9, 3)
    assert torch.allclose(gaussian_map.get_sh_coefficients[:, 1:, :], torch.full((2, 8, 3), 0.01), atol=1.0e-6)
    assert gaussian_map._anchor_source_window_id.tolist() == [3, 3]
    assert gaussian_map._anchor_source_frame_start.tolist() == [0, 0]
    assert gaussian_map._anchor_source_frame_end.tolist() == [3, 3]


def test_resplat_state_fusion_merges_nearby_gaussians() -> None:
    gaussian_map = PanoGaussianMap(config={"BackendOptimization": {"sh_degree": 2}}, device="cpu")
    first = _state(torch.tensor([[0.0, 0.0, 1.0]]))
    second = _state(torch.tensor([[0.03, 0.0, 1.0]]))
    cfg = {"voxel_size": 0.1, "merge_radius": 0.08, "min_confidence": 0.0, "min_opacity": 0.0}

    gaussian_map.add_or_fuse_resplat_gaussians(first, frame_ids=[0, 1, 2, 3], config=cfg)
    stats = gaussian_map.add_or_fuse_resplat_gaussians(second, frame_ids=[4, 5, 6, 7], config=cfg)

    assert stats["fused"] == 1
    assert stats["inserted"] == 0
    assert gaussian_map.anchor_count() == 1
    assert 0.0 < float(gaussian_map.get_xyz.detach()[0, 0]) < 0.03
    assert int(gaussian_map._anchor_obs_count[0]) == 2


def test_resplat_state_fusion_runs_twenty_global_iters(monkeypatch) -> None:
    gaussian_map = PanoGaussianMap(config={"BackendOptimization": {"sh_degree": 2}}, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map)
    captured: dict[str, int] = {}

    def fake_optimize_feedforward_window(**kwargs):
        captured["steps"] = int(mapper.optim_cfg["FeedForwardWindow"]["steps"])
        captured["pose_refine"] = int(bool(mapper.optim_cfg.get("pose_refine_enable", True)))
        return {"loss": 0.0, "steps": float(captured["steps"])}

    monkeypatch.setattr(mapper, "optimize_feedforward_window", fake_optimize_feedforward_window)
    metrics = mapper.optimize_resplat_global_window(frame_ids=[0, 1, 2, 3], iters=20)

    assert captured["steps"] == 20
    assert captured["pose_refine"] == 0
    assert metrics["resplat_global_configured_steps"] == 20.0
