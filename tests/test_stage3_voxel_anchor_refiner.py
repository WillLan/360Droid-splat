from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import yaml

from backend.pano_gs.mapper import PanoGaussianMap
from backend.pano_gs.stage2_global_fusion import Stage2GlobalMapFusion
from frontend.spherical_selfi.window_packet import LocalGaussianWindowPacket
from geometry.sim3 import sim3_from_components, sim3_identity
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead
from models.spherical_voxel_anchor_refiner import (
    AnchorErrorPoolOutput,
    BinaryAnchorErrorPooler,
    SimplifiedVoxelAnchorRefiner,
    VoxelAnchorConfig,
    VoxelAnchorRenderGroup,
    VoxelAnchorStage3Model,
    _chunked_eigh_3x3,
    depth_to_voxel_level,
    depth_to_voxel_size,
    load_voxel_anchor_checkpoint,
    voxelize_per_pixel_gaussians,
)
from training.train_spherical_voxel_anchor_refiner import (
    VoxelAnchorWindowResult,
    VoxelAnchorTrainableModel,
    _detach_validation_visualization,
    _validate,
    save_checkpoint,
)


def _observation(*, views: int = 4, height: int = 8, width: int = 16):
    torch.manual_seed(31)
    features = torch.randn(1, views, 24, height, width)
    images = torch.rand(1, views, 3, height, width)
    depth = torch.full((1, views, 1, height, width), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    poses[0, :, 0, 3] = torch.arange(views).float() * 0.05
    head = SphericalSelfiGaussianHead(
        channels=(8, 16, 24, 32),
        mlp_hidden_dim=16,
    )
    return head(features, images, depth, poses), features, images


def _single_member_anchor(*, views: int = 4):
    observation, features, images = _observation(views=views)
    valid = torch.zeros_like(observation.valid_mask)
    valid[:, 0, :, 3, 5] = True
    observation = replace(observation, valid_mask=valid)
    config = VoxelAnchorConfig(use_resnet_error=False, pretrained_resnet=False)
    anchors = voxelize_per_pixel_gaussians(
        observation,
        features,
        images,
        config,
        valid_mask=valid,
    )
    assert anchors.num_anchors == 1
    return observation, features, images, anchors


def _constant_feedback(anchors, values, alphas, target_valid=None):
    batch, views = anchors.batch_size, anchors.num_views
    height, width = anchors.image_size
    error = anchors.xyz.new_zeros(batch, views, 32, max(1, height // 4), max(1, width // 4))
    depth = anchors.xyz.new_zeros(batch, views, 1, height, width)
    alpha = anchors.xyz.new_zeros(batch, views, 1, height, width)
    for view, (value, opacity) in enumerate(zip(values, alphas)):
        error[:, view] = float(value)
        camera = anchors.local_poses_c2w[0, view]
        distance = torch.linalg.norm(anchors.xyz[0] - camera[:3, 3])
        depth[:, view] = distance
        alpha[:, view] = float(opacity)
    visibility = torch.ones(batch, views, anchors.num_anchors, dtype=torch.bool)
    if target_valid is None:
        target_valid = torch.ones_like(depth, dtype=torch.bool)
    return error, VoxelAnchorRenderGroup(
        rendered=torch.zeros(batch, views, 3, height, width),
        depth=depth,
        alpha=alpha,
        anchor_visibility=visibility,
        profiles={},
    ), target_valid


def test_validation_visualization_is_detached_on_cpu() -> None:
    target = torch.rand(1, 4, 3, 8, 16, requires_grad=True)
    rendered = target * 0.5
    result = VoxelAnchorWindowResult(
        final=None,  # type: ignore[arg-type]
        ba0_observation=None,  # type: ignore[arg-type]
        snapshots={},
        rendered_snapshots={"voxelized": rendered, "refine3": rendered + 0.1},
        metrics={},
    )
    visualization = _detach_validation_visualization(result, target)
    assert visualization.target.device.type == "cpu"
    assert not visualization.target.requires_grad
    for snapshot in visualization.rendered_snapshots.values():
        assert snapshot.device.type == "cpu"
        assert snapshot.dtype == torch.float32
        assert not snapshot.requires_grad


def test_validation_runs_without_grad_and_releases_batch_temporaries() -> None:
    class Head(torch.nn.Module):
        def forward(self, features, images, initial_depth, poses, *, frame_ids):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(
                refined_depth=torch.ones(1, 1, 1, 2, 4),
                valid_mask=torch.ones(1, 1, 1, 2, 4, dtype=torch.bool),
            )

    model = torch.nn.Linear(1, 1)
    model.train()
    images = torch.rand(1, 1, 3, 2, 4)
    extracted = (
        torch.rand(1, 1, 24, 2, 4),
        images,
        torch.ones(1, 1, 1, 2, 4),
        torch.eye(4).view(1, 1, 4, 4),
    )
    rendered = torch.rand(1, 1, 3, 2, 4)
    result = VoxelAnchorWindowResult(
        final=None,  # type: ignore[arg-type]
        ba0_observation=None,  # type: ignore[arg-type]
        snapshots={},
        rendered_snapshots={"voxelized": rendered, "refine3": rendered},
        metrics={
            "stage3/rgb_l1": 0.2,
            "stage3/relative_ba0_depth": 0.3,
        },
    )

    def run_without_grad(*args, **kwargs):
        assert not torch.is_grad_enabled()
        assert kwargs["backward"] is False
        return result

    config = {
        "image": {"head_height": 2, "head_width": 4},
        "train": {"amp": False, "max_val_batches": 1},
    }
    batch = {"images": images, "frame_ids": torch.zeros(1, 1, dtype=torch.long)}
    with (
        patch(
            "training.train_spherical_voxel_anchor_refiner.extract_frozen_inputs",
            return_value=extracted,
        ),
        patch(
            "training.train_spherical_voxel_anchor_refiner._build_match_cache",
            return_value=object(),
        ),
        patch(
            "training.train_spherical_voxel_anchor_refiner.run_voxel_anchor_window",
            side_effect=run_without_grad,
        ),
        patch(
            "training.train_spherical_voxel_anchor_refiner._render_metrics",
            return_value={"final_psnr": 12.0},
        ),
        patch("training.train_spherical_voxel_anchor_refiner._empty_cuda_cache") as empty_cache,
    ):
        metrics, visualization = _validate(
            model,  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            Head(),
            object(),  # type: ignore[arg-type]
            object(),
            [batch],  # type: ignore[arg-type]
            config,
            feature_device=torch.device("cpu"),
            train_device=torch.device("cpu"),
            step=1000,
        )
    assert model.training
    assert metrics["val/final_psnr"] == 12.0
    assert visualization is not None
    assert visualization.target.device.type == "cpu"
    assert not visualization.target.requires_grad
    assert empty_cache.call_count == 2


def test_depth_boundaries_select_the_coarser_level_exactly() -> None:
    config = VoxelAnchorConfig(use_resnet_error=False, pretrained_resnet=False)
    depth = torch.tensor([4.999, 5.0, 19.999, 20.0, 39.999, 40.0])
    assert depth_to_voxel_level(depth, config).tolist() == [0, 1, 1, 2, 2, 3]
    torch.testing.assert_close(
        depth_to_voxel_size(depth, config),
        torch.tensor([0.04, 0.08, 0.08, 0.16, 0.16, 0.32]),
    )


def test_voxel_config_rejects_invalid_boundaries_and_sizes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        VoxelAnchorConfig(depth_boundaries=(5.0, 5.0, 40.0))
    with pytest.raises(ValueError, match="four positive"):
        VoxelAnchorConfig(voxel_sizes=(0.04, 0.08, -0.16, 0.32))


def test_chunked_anchor_eigh_matches_direct_decomposition() -> None:
    torch.manual_seed(8)
    matrix = torch.randn(23, 3, 3)
    covariance = matrix @ matrix.transpose(-1, -2) + 1.0e-4 * torch.eye(3)
    expected_values, expected_vectors = torch.linalg.eigh(covariance)
    values, vectors = _chunked_eigh_3x3(covariance, chunk_size=4)
    torch.testing.assert_close(values, expected_values)
    reconstructed = vectors @ torch.diag_embed(values) @ vectors.transpose(-1, -2)
    torch.testing.assert_close(reconstructed, covariance, atol=2.0e-5, rtol=2.0e-5)


def test_nonfinite_member_attributes_are_filtered_before_moment_matching() -> None:
    observation, features, images = _observation(views=1)
    valid = torch.zeros_like(observation.valid_mask)
    valid[:, 0, :, 3, 5] = True
    log_scale = observation.log_scale_multiplier.clone()
    log_scale[:, 0, 0, 3, 5] = torch.nan
    observation = replace(observation, valid_mask=valid, log_scale_multiplier=log_scale)
    anchors = voxelize_per_pixel_gaussians(
        observation,
        features,
        images,
        VoxelAnchorConfig(use_resnet_error=False, pretrained_resnet=False),
    )
    assert anchors.num_anchors == 0


def test_same_point_from_different_source_depths_has_one_reference_key() -> None:
    observation, features, images = _observation(views=2)
    valid = torch.zeros_like(observation.valid_mask)
    row, column = 3, 5
    valid[:, :, :, row, column] = True
    ray = observation.source_ray[row, column]
    poses = observation.poses_c2w.clone()
    poses[0, 1, :3, 3] = 0.5 * ray
    depth = observation.refined_depth.clone()
    depth[0, 0, 0, row, column] = 2.0
    depth[0, 1, 0, row, column] = 1.5
    observation = replace(
        observation,
        refined_depth=depth,
        depth_residual=depth - observation.initial_depth,
        poses_c2w=poses,
        valid_mask=valid,
    )
    config = VoxelAnchorConfig(use_resnet_error=False, pretrained_resnet=False)
    anchors = voxelize_per_pixel_gaussians(observation, features, images, config)
    assert anchors.num_anchors == 1
    torch.testing.assert_close(
        anchors.membership.reference_depth,
        torch.tensor([2.0, 2.0]),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert anchors.level.tolist() == [0]
    assert anchors.membership.anchor_index.tolist() == [0, 0]

    changed_scale = replace(
        observation,
        log_scale_multiplier=torch.full_like(observation.log_scale_multiplier, -4.0),
    )
    changed = voxelize_per_pixel_gaussians(changed_scale, features, images, config)
    assert torch.equal(changed.level, anchors.level)
    assert torch.equal(changed.grid_coord, anchors.grid_coord)


def test_binary_multiview_pooling_keeps_only_signed_abs_and_coverage() -> None:
    _, _, _, anchors = _single_member_anchor()
    pooler = BinaryAnchorErrorPooler()
    target_valid = torch.ones(1, 4, 1, *anchors.image_size, dtype=torch.bool)
    target_valid[:, 3] = False
    error, render, target_valid = _constant_feedback(
        anchors,
        values=[1.0, -2.0, 100.0, 7.0],
        alphas=[0.9, 0.051, 0.05, 0.9],
        target_valid=target_valid,
    )
    output = pooler(anchors, error, render, target_valid)
    assert output.raw_statistics.shape == (1, 73)
    assert output.feature.shape == (1, 32)
    torch.testing.assert_close(output.raw_statistics[:, :32], torch.full((1, 32), -0.5))
    torch.testing.assert_close(output.raw_statistics[:, 32:64], torch.full((1, 32), 1.5))
    torch.testing.assert_close(output.raw_statistics[:, 64:65], torch.tensor([[0.5]]))

    # Alpha is a strict binary gate; changing valid alpha magnitudes does not
    # alter either view's relative weight.
    render.alpha[:, 0] = 0.051
    render.alpha[:, 1] = 0.99
    changed = pooler(anchors, error, render, target_valid)
    torch.testing.assert_close(changed.raw_statistics, output.raw_statistics)


def test_binary_pooling_accumulates_bfloat16_error_samples_in_float32() -> None:
    _, _, _, anchors = _single_member_anchor()
    pooler = BinaryAnchorErrorPooler()
    error, render, target_valid = _constant_feedback(
        anchors,
        values=[1.0, -2.0, 3.0, -4.0],
        alphas=[0.9, 0.9, 0.9, 0.9],
    )

    def sample_first_pixel(feature_map: torch.Tensor, pixel: torch.Tensor) -> torch.Tensor:
        return feature_map[:, 0, 0].view(1, -1).expand(int(pixel.shape[0]), -1)

    with patch(
        "models.spherical_voxel_anchor_refiner.sample_erp_with_wrap",
        side_effect=sample_first_pixel,
    ):
        output = pooler(anchors, error.bfloat16(), render, target_valid)
    assert output.raw_statistics.dtype == torch.float32
    assert bool(torch.isfinite(output.raw_statistics).all())


def test_target_order_permutation_leaves_anchor_statistics_unchanged() -> None:
    _, _, _, anchors = _single_member_anchor()
    pooler = BinaryAnchorErrorPooler()
    error, render, valid = _constant_feedback(
        anchors,
        values=[1.0, -2.0, 3.0, -4.0],
        alphas=[0.9, 0.8, 0.7, 0.6],
    )
    original = pooler(anchors, error, render, valid)
    permutation = torch.tensor([2, 0, 3, 1])
    permuted_anchors = replace(
        anchors,
        local_poses_c2w=anchors.local_poses_c2w[:, permutation],
        frame_ids=anchors.frame_ids[:, permutation],
    )
    permuted_render = replace(
        render,
        rendered=render.rendered[:, permutation],
        depth=render.depth[:, permutation],
        alpha=render.alpha[:, permutation],
        anchor_visibility=render.anchor_visibility[:, permutation],
    )
    permuted = pooler(
        permuted_anchors,
        error[:, permutation],
        permuted_render,
        valid[:, permutation],
    )
    torch.testing.assert_close(permuted.raw_statistics, original.raw_statistics, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(permuted.feature, original.feature, atol=1e-6, rtol=1e-6)


def test_no_feedback_is_exact_identity_and_position_stays_inside_voxel() -> None:
    _, _, _, anchors = _single_member_anchor()
    identity_refiner = SimplifiedVoxelAnchorRefiner(anchors.config)
    identity_error = AnchorErrorPoolOutput(
        feature=torch.zeros(1, 32),
        raw_statistics=torch.zeros(1, 73),
        has_feedback=torch.ones(1, dtype=torch.bool),
        coverage=torch.ones(1, 1),
    )
    identity = identity_refiner(anchors, identity_error, iteration_index=0)
    for field in ("xyz", "rotation", "log_scales", "sh_coefficients", "opacity_logit"):
        torch.testing.assert_close(getattr(identity.observation, field), getattr(anchors, field))

    refiner = SimplifiedVoxelAnchorRefiner(anchors.config)
    assert refiner.static_encoder[0].in_features == 27
    assert refiner.state_encoder[0].in_features == 38
    with torch.no_grad():
        refiner.geometry_head[-1].bias.fill_(10.0)
        refiner.appearance_head[-1].bias.fill_(10.0)
    no_feedback = AnchorErrorPoolOutput(
        feature=torch.randn(1, 32),
        raw_statistics=torch.zeros(1, 73),
        has_feedback=torch.zeros(1, dtype=torch.bool),
        coverage=torch.zeros(1, 1),
    )
    unchanged = None
    no_feedback_current = anchors
    no_feedback_hidden = None
    for iteration in range(3):
        unchanged = refiner(
            no_feedback_current,
            no_feedback,
            iteration_index=iteration,
            hidden=no_feedback_hidden,
        )
        no_feedback_current = unchanged.observation
        no_feedback_hidden = unchanged.hidden
    assert unchanged is not None
    for field in ("xyz", "rotation", "log_scales", "sh_coefficients", "opacity_logit"):
        assert torch.equal(getattr(unchanged.observation, field), getattr(anchors, field))

    feedback = replace(no_feedback, has_feedback=torch.ones(1, dtype=torch.bool))
    current = anchors
    hidden = None
    for iteration in range(3):
        result = refiner(current, feedback, iteration_index=iteration, hidden=hidden)
        current, hidden = result.observation, result.hidden
    lower = current.voxel_center - 0.5 * current.voxel_size
    upper = current.voxel_center + 0.5 * current.voxel_size
    assert bool(((current.xyz >= lower) & (current.xyz <= upper)).all())
    assert bool((current.scaling >= current.min_scales - 1.0e-8).all())
    rescaled = current.rescale_geometry(0.01)
    torch.testing.assert_close(rescaled.voxel_size, current.voxel_size)
    assert bool((rescaled.scaling >= rescaled.min_scales - 1.0e-8).all())


def test_backend_preserves_depth_selected_level_after_fusion_and_correction() -> None:
    observation, features, images, anchors = _single_member_anchor(views=1)
    anchors = anchors.detach_for_backend()
    assert anchors.membership.anchor_index.numel() == 0
    packet = LocalGaussianWindowPacket.from_observation(
        window_id=0,
        observation=observation,
        adapter_features=features,
        frame_ids=(0,),
        verification_size=observation.image_size,
        anchor_observation=anchors,
    )
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    fusion = Stage2GlobalMapFusion(
        gaussian_map,
        voxel_sizes=(0.04, 0.08, 0.16, 0.32),
        min_confidence=0.0,
        min_opacity=0.0,
    )
    fusion.fuse_packet(packet, sim3_identity())
    assert gaussian_map._anchor_level.tolist() == [0]
    torch.testing.assert_close(gaussian_map._anchor_voxel_size, torch.tensor([0.04]))
    correction = sim3_from_components(1.0, torch.eye(3), torch.tensor([0.2, 0.0, 0.0]))
    fusion.apply_owner_corrections({0: sim3_identity()}, {0: correction})
    assert gaussian_map._anchor_level.tolist() == [0]
    torch.testing.assert_close(gaussian_map._anchor_voxel_size, torch.tensor([0.04]))


def test_formal_runtime_config_keeps_new_path_disabled_by_default() -> None:
    path = Path(__file__).parents[1] / "configs" / "spherical_selfi_global_gs_slam.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    voxel = config["VoxelAnchorRefiner"]
    assert voxel["enabled"] is False
    assert voxel["depth_boundaries"] == [5.0, 20.0, 40.0]
    assert voxel["voxel_sizes"] == [0.04, 0.08, 0.16, 0.32]


def test_training_checkpoint_loads_directly_into_runtime_model(tmp_path: Path) -> None:
    config = VoxelAnchorConfig(use_resnet_error=False, pretrained_resnet=False)
    training_model = VoxelAnchorTrainableModel(config)
    optimizer = torch.optim.AdamW(training_model.parameters(), lr=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    checkpoint = save_checkpoint(
        tmp_path / "anchor.pt",
        model=training_model,
        optimizer=optimizer,
        scheduler=scheduler,
        config={"VoxelAnchorRefiner": {"enabled": True}},
        step=7,
        metrics={"val/final_psnr": 20.0},
        adapter_sha256="synthetic",
        stage2_checkpoint_sha256=None,
    )
    runtime_model = VoxelAnchorStage3Model(config)
    payload = load_voxel_anchor_checkpoint(checkpoint, model=runtime_model)
    assert payload["format"] == "spherical_voxel_anchor_refiner_v1"
    assert payload["global_step"] == 7
    for expected, actual in zip(training_model.parameters(), runtime_model.parameters()):
        torch.testing.assert_close(actual, expected)
