from __future__ import annotations

import torch
from torch import nn

from frontend.pano_vggt.pano_resplat_feedback import PanoViewPoseResidualHead
from frontend.pano_vggt.pano_resplat_frontend import PanoReSplatFrontend
from frontend.pano_vggt.pano_resplat_refiner import PanoGaussianUpdateBlock
from frontend.pano_vggt.pano_resplat_renderer import PanoGaussianRendererAdapter
from frontend.pano_vggt.pano_resplat_voxel import VoxelGaussianCompactor
from frontend.pano_vggt.resplat_types import PanoGaussianState, state_to_explicit_gaussian_set
from frontend.pano_vggt.train_resplat_gaussian import _lpips_resize_hw


def _dense_state(batch: int = 1, *, requires_grad: bool = False) -> PanoGaussianState:
    means = torch.tensor(
        [
            [
                [0.001, 0.001, 2.001],
                [0.009, 0.003, 2.004],
                [0.041, 0.001, 2.001],
                [0.049, 0.002, 2.002],
                [0.090, 0.001, 2.001],
            ]
        ],
        dtype=torch.float32,
    ).repeat(batch, 1, 1)
    if requires_grad:
        means.requires_grad_(True)
    log_scales = torch.log(torch.tensor([0.01, 0.02, 0.03], dtype=torch.float32)).view(1, 1, 3).repeat(batch, 5, 1)
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    ).view(1, 5, 4).repeat(batch, 1, 1)
    opacity = torch.tensor([0.2, 0.8, 0.0, 0.4, -1.0], dtype=torch.float32).view(1, 5, 1).repeat(batch, 1, 1)
    sh = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1).repeat(batch, 1, 3, 4) * 0.01
    latent = torch.ones(batch, 5, 6)
    source_ids = torch.tensor([[0, 1, 2, 3, 1]], dtype=torch.long).repeat(batch, 1)
    source_uv = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]], dtype=torch.float32).repeat(batch, 1, 1)
    valid = torch.ones(batch, 5, dtype=torch.bool)
    confidence = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3], dtype=torch.float32).view(1, 5, 1).repeat(batch, 1, 1)
    return PanoGaussianState(
        means=means,
        log_scales=log_scales,
        rotations_unnorm=rotations,
        opacity_logits=opacity,
        sh_coeffs=sh,
        latent_features=latent,
        source_view_ids=source_ids,
        source_uv=source_uv,
        valid_mask=valid,
        confidence=confidence,
    )


def test_voxel_compactor_fuses_occupied_voxels_and_keeps_dominant_source():
    state = _dense_state()
    compactor = VoxelGaussianCompactor(voxel_size=0.02, detach_input=True)

    compact, stats = compactor.compact(state)

    assert compact.valid_mask.sum().item() == 3
    assert compact.means.shape[1] == 3
    assert torch.allclose(compact.means[0, 0], state.means[0, :2].mean(dim=0), atol=1.0e-6)
    assert compact.source_view_ids[0, 0].item() == 1
    assert torch.allclose(compact.source_uv[0, 0], torch.tensor([3.0, 4.0]))
    assert compact.latent_features[0, 0, 0] > 0.0
    assert torch.allclose(compact.latent_features[0, 0, 1], compact.confidence[0, 0, 0])
    assert stats["dense_count"].item() == 5.0
    assert stats["anchor_count"].item() == 3.0
    assert torch.allclose(stats["compression_ratio"], torch.tensor(0.6))


def test_voxel_compactor_detaches_initializer_outputs_by_default():
    state = _dense_state(requires_grad=True)
    compactor = VoxelGaussianCompactor(voxel_size=0.02, detach_input=True)

    compact = compactor(state)

    assert not compact.means.requires_grad


def test_voxel_compactor_caps_anchor_count_by_confidence():
    state = _dense_state()
    compactor = VoxelGaussianCompactor(voxel_size=0.02, max_anchors=2, detach_input=True)

    compact, stats = compactor.compact(state)

    assert compact.valid_mask.sum().item() == 2
    assert compact.means.shape[1] == 2
    assert stats["anchor_count"].item() == 2.0
    assert torch.all(compact.confidence[0, compact.valid_mask[0], 0] >= 0.3)


def test_voxel_compactor_preserves_quaternion_materialization_and_render_smoke():
    state = _dense_state()
    compact = VoxelGaussianCompactor(voxel_size=0.02)(state)

    explicit = state_to_explicit_gaussian_set(compact, 0)
    norms = torch.linalg.norm(explicit.get_rotation, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1.0e-5)

    poses = torch.eye(4).view(1, 4, 4)
    render = PanoGaussianRendererAdapter(soft_max_points=16).render_state(compact, poses, (8, 16), renderer_backend="soft_splat")
    assert render.color.shape == (1, 3, 8, 16)
    assert torch.isfinite(render.color).all()


def test_update_block_consumes_compact_anchor_state():
    state = VoxelGaussianCompactor(voxel_size=0.02)(_dense_state())
    block = PanoGaussianUpdateBlock(
        feedback_dim=5,
        latent_dim=state.latent_dim,
        sh_dim=state.sh_dim,
        hidden_dim=16,
        knn=2,
        num_heads=1,
        attn_proj_channels=8,
        num_basic_refine_blocks=2,
        max_knn_points=0,
    )
    feedback = torch.randn(state.batch_size, state.num_gaussians, 5)

    updated, metrics = block(state, feedback)

    assert updated.means.shape == state.means.shape
    assert torch.isfinite(updated.means).all()
    assert "mean_delta_abs" in metrics


class _FakeInitializer(nn.Module):
    state_dim = 6
    sh_dim = 4

    def __init__(self, state: PanoGaussianState) -> None:
        super().__init__()
        self.state = state

    def forward(self, *_args, **_kwargs) -> PanoGaussianState:
        return self.state


def test_frontend_records_dense_and_compact_init_states():
    dense = _dense_state()
    frontend = PanoReSplatFrontend(
        initializer=_FakeInitializer(dense),
        update_block=PanoGaussianUpdateBlock(feedback_dim=32, latent_dim=dense.latent_dim, sh_dim=dense.sh_dim, hidden_dim=8, num_heads=1),
        compactor=VoxelGaussianCompactor(voxel_size=0.02),
        renderer=PanoGaussianRendererAdapter(soft_max_points=16),
    )
    context = {
        "images": torch.zeros(1, 1, 3, 8, 16),
        "features": torch.zeros(1, 1, 4, 2, 4),
        "depths": torch.ones(1, 1, 1, 8, 16),
        "poses_c2w": torch.eye(4).view(1, 1, 4, 4),
        "valid_mask": torch.ones(1, 1, 1, 8, 16, dtype=torch.bool),
    }

    out = frontend(context, num_refine=0)

    assert out["dense_init_state"].num_gaussians == 5
    assert out["init_state"].num_gaussians == 3
    assert out["compactor_debug"]["anchor_count"].item() == 3.0


def test_lpips_full_resolution_config_keeps_512x1024_inputs():
    cfg = {"lpips_resize_height": 512, "lpips_resize_width": 1024}

    assert _lpips_resize_hw(cfg, (512, 1024)) == (512, 1024)


def test_view_pose_residual_head_updates_poses_with_bounds():
    head = PanoViewPoseResidualHead(4, hidden_dim=8, max_rotation_deg=1.0, max_translation=0.03)
    with torch.no_grad():
        head.net[-1].bias.fill_(10.0)
    tokens = torch.ones(1, 2, 4)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)

    refined, metrics = head(tokens, poses)

    assert refined.shape == poses.shape
    assert metrics["pose_rot_deg_abs"] <= 1.0 + 1.0e-5
    assert metrics["pose_trans_norm"] <= (3.0**0.5) * 0.03 + 1.0e-5
    assert not torch.allclose(refined, poses)


def test_view_pose_residual_head_has_nonzero_zero_init_gradient():
    head = PanoViewPoseResidualHead(4, hidden_dim=8, max_rotation_deg=1.0, max_translation=0.03)
    tokens = torch.ones(1, 2, 4)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)

    refined, _metrics = head(tokens, poses)
    loss = refined[..., :3, 3].sum() + refined[..., :3, :3].sum()
    loss.backward()

    assert head.net[-1].bias.grad is not None
    assert torch.isfinite(head.net[-1].bias.grad).all()
    assert head.net[-1].bias.grad.abs().sum() > 0.0
