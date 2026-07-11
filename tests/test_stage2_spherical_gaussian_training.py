from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from backend.pano_gs.adapter import _optional_gsplat360
from data.stage2_source_reconstruction_dataset import Stage2SourceReconstructionDataset
from frontend.pano_droid.interfaces import FrontendOutput
from losses.spherical_gaussian_render_loss import (
    periodic_ssim_map,
    spherical_pseudo_geometry_consistency_loss,
    spherical_weighted_l1,
    stage2_gaussian_render_loss,
)
from models.spherical_selfi_gaussian_head import SphericalSelfiGaussianHead
from training.train_spherical_selfi_gaussian_head import (
    DistributedContext,
    _effective_batch_size,
    _is_interval_step,
    _scheduler,
    build_frozen_feature_stack,
    default_config,
    load_stage2_checkpoint,
    render_observation_target,
    render_observation_views,
    save_stage2_checkpoint,
    train_spherical_selfi_gaussian_head,
)


def _small_head() -> SphericalSelfiGaussianHead:
    return SphericalSelfiGaussianHead(channels=(8, 16, 24, 32), mlp_hidden_dim=16)


def test_periodic_ssim_treats_wrapped_roll_consistently() -> None:
    image = torch.rand(1, 3, 8, 16)
    shifted = torch.roll(image, 1, dims=-1)
    map_a = periodic_ssim_map(image, shifted)
    map_b = periodic_ssim_map(torch.roll(image, 5, dims=-1), torch.roll(shifted, 5, dims=-1))
    torch.testing.assert_close(map_b, torch.roll(map_a, 5, dims=-1), atol=1e-6, rtol=1e-5)


def test_latitude_weighted_l1_downweights_equal_polar_error() -> None:
    target = torch.zeros(3, 8, 16)
    equator = target.clone()
    pole = target.clone()
    equator[:, 4, :] = 1.0
    pole[:, 0, :] = 1.0
    assert spherical_weighted_l1(equator, target) > spherical_weighted_l1(pole, target)


def test_pseudo_correspondence_geometry_loss_uses_refined_depth() -> None:
    head = _small_head()
    height, width = 8, 16
    feature = torch.zeros(1, 2, 24, height, width)
    image = torch.zeros(1, 2, 3, height, width)
    depth = torch.full((1, 2, 1, height, width), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, 2, 1, 1)
    observation = head(feature, image, depth, poses)
    consistent = spherical_pseudo_geometry_consistency_loss(
        observation, batch_index=0, num_query_per_pair=32, min_depth=0.1, max_depth=10.0
    )
    assert float(consistent.detach()) < 1.0e-6
    changed_depth = observation.refined_depth.clone()
    changed_depth[:, 1] *= 1.2
    inconsistent = spherical_pseudo_geometry_consistency_loss(
        observation.with_geometry(refined_depth=changed_depth),
        batch_index=0,
        num_query_per_pair=32,
        min_depth=0.1,
        max_depth=10.0,
    )
    assert inconsistent > consistent


def test_stage2_checkpoint_round_trip_and_adapter_sha_guard(tmp_path: Path) -> None:
    head = _small_head()
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4)
    scheduler = _scheduler(optimizer, warmup_steps=2, max_steps=10)
    path = save_stage2_checkpoint(
        tmp_path / "head.pt",
        head=head,
        config={"panovggt": {"class_path": "fake"}},
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        metrics={"loss": 1.0},
        adapter_sha256="abc",
        best_val_psnr=12.0,
    )
    restored = _small_head()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=2e-4)
    restored_scheduler = _scheduler(restored_optimizer, warmup_steps=2, max_steps=10)
    step, metrics, best = load_stage2_checkpoint(
        path,
        head=restored,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_adapter_sha256="abc",
    )
    assert step == 3 and metrics == {"loss": 1.0} and best == 12.0
    for expected, actual in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="SHA256"):
        load_stage2_checkpoint(path, head=restored, expected_adapter_sha256="wrong")


def test_synthetic_panovggt_and_adapter_stack_is_frozen() -> None:
    config = default_config()
    wrapper, adapter, _, _ = build_frozen_feature_stack(config, device=torch.device("cpu"))
    assert not wrapper.training and not adapter.training
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())
    assert all(not parameter.requires_grad for parameter in adapter.parameters())


def test_ddp_effective_batch_size_uses_microbatch_accumulation_and_world_size() -> None:
    train_config = {"batch_size": 1, "gradient_accumulation_steps": 2}
    distributed = DistributedContext(enabled=True, rank=0, local_rank=0, world_size=2)
    assert _effective_batch_size(train_config, distributed) == 4
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        _effective_batch_size({"batch_size": 1, "gradient_accumulation_steps": 0}, distributed)


def test_diagnostics_interval_runs_only_every_200_steps() -> None:
    assert not _is_interval_step(1, 200)
    assert not _is_interval_step(199, 200)
    assert _is_interval_step(200, 200)
    assert _is_interval_step(400, 200)
    with pytest.raises(ValueError, match="positive"):
        _is_interval_step(200, 0)


class _DifferentiableFakeRenderer:
    def render(self, camera, gaussians) -> dict[str, torch.Tensor]:
        height = int(camera.image_height)
        width = int(camera.image_width)
        opacity = gaussians.get_opacity
        color = (gaussians.get_features * opacity).mean(dim=0)
        geometry = 1.0e-3 * (
            gaussians.get_xyz.mean()
            + gaussians.get_scaling.mean()
            + gaussians.get_rotation.mean()
        )
        render = (color + geometry).view(3, 1, 1).expand(3, height, width)
        alpha = opacity.mean().view(1, 1, 1).expand(1, height, width)
        depth = torch.linalg.norm(gaussians.get_xyz, dim=-1).mean().view(1, 1, 1).expand(1, height, width)
        return {"render": render, "alpha": alpha, "depth": depth}


def test_joint_multiview_backward_matches_recomputed_head_gradients() -> None:
    torch.manual_seed(7)
    repeated_head = _small_head()
    joint_head = copy.deepcopy(repeated_head)
    height, width, views = 8, 16, 4
    feature = torch.randn(1, views, 24, height, width)
    image = torch.rand(1, views, 3, height, width)
    depth = torch.full((1, views, 1, height, width), 2.0)
    poses = torch.eye(4).view(1, 1, 4, 4).repeat(1, views, 1, 1)
    renderer = _DifferentiableFakeRenderer()

    repeated_forward_count = 0
    joint_forward_count = 0

    def count_repeated(*_args) -> None:
        nonlocal repeated_forward_count
        repeated_forward_count += 1

    def count_joint(*_args) -> None:
        nonlocal joint_forward_count
        joint_forward_count += 1

    repeated_hook = repeated_head.register_forward_hook(count_repeated)
    joint_hook = joint_head.register_forward_hook(count_joint)
    repeated_loss = torch.zeros(())
    for target_view in range(views):
        observation = repeated_head(feature, image, depth, poses)
        package = render_observation_target(
            renderer,
            observation,
            batch_index=0,
            target_view=target_view,
        )
        terms = stage2_gaussian_render_loss(
            [package],
            image[0, target_view : target_view + 1],
            observation,
        )
        repeated_loss = repeated_loss + terms["loss"] / float(views)
    repeated_loss.backward()

    joint_observation = joint_head(feature, image, depth, poses)
    joint_packages = render_observation_views(renderer, joint_observation, batch_index=0)
    joint_terms = stage2_gaussian_render_loss(
        joint_packages,
        image[0],
        joint_observation,
    )
    joint_terms["loss"].backward()
    repeated_hook.remove()
    joint_hook.remove()

    assert repeated_forward_count == views
    assert joint_forward_count == 1
    torch.testing.assert_close(joint_terms["loss"], repeated_loss, atol=1.0e-6, rtol=1.0e-5)
    for repeated_parameter, joint_parameter in zip(repeated_head.parameters(), joint_head.parameters()):
        if repeated_parameter.grad is None or joint_parameter.grad is None:
            assert repeated_parameter.grad is None and joint_parameter.grad is None
        else:
            torch.testing.assert_close(joint_parameter.grad, repeated_parameter.grad, atol=2.0e-5, rtol=2.0e-4)


def test_source_dataset_has_no_target_and_reproducible_random_stride(tmp_path: Path) -> None:
    records = []
    for frame in range(16):
        image_path = tmp_path / f"frame_{frame:03d}.png"
        Image.new("RGB", (16, 8), (frame, frame, frame)).save(image_path)
        records.append(
            {
                "scene_id": "scene",
                "sequence_id": "sequence",
                "frame_id": str(frame),
                "timestamp": float(frame),
                "rgb_path": image_path.name,
                "split": "train",
                "domain": "outdoor",
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(records), encoding="utf-8")
    dataset = Stage2SourceReconstructionDataset(
        manifest,
        split="train",
        views_per_sample=4,
        stride_min=2,
        stride_max=4,
        image_height=8,
        image_width=16,
        seed=99,
    )
    dataset.set_epoch(3)
    first = dataset[0]
    repeated = dataset[0]
    assert "target" not in first and "target_image" not in first
    assert first["images"].shape == (4, 3, 8, 16)
    assert first["stride"] == repeated["stride"]
    torch.testing.assert_close(first["frame_ids"], repeated["frame_ids"])


def test_stage2_is_disabled_by_default_and_frontend_output_contract_is_unchanged() -> None:
    config = default_config()
    assert config["stage2"]["enabled"] is False
    assert set(FrontendOutput.__dataclass_fields__) == {
        "frame_id",
        "timestamp",
        "pose_c2w",
        "relative_pose",
        "pose_confidence",
        "inverse_depth",
        "depth_confidence",
        "spherical_flow",
        "keyframe_score",
        "is_keyframe",
        "ba_residual",
        "tracking_status",
        "world_points",
        "world_points_confidence",
        "valid_world_points_mask",
    }


def test_stage2_modules_do_not_import_removed_frontend_families() -> None:
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "models/per_pixel_gaussian_observation.py",
        root / "models/spherical_selfi_gaussian_head.py",
        root / "data/stage2_source_reconstruction_dataset.py",
        root / "losses/spherical_gaussian_render_loss.py",
        root / "training/train_spherical_selfi_gaussian_head.py",
    ]
    forbidden = ("anchor", "voxel", "recurrent", "point_transformer", "frontend.pano_vggt.gaussian_head")
    imported: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
    assert not [name for name in imported if any(token in name.lower() for token in forbidden)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Stage 2 renderer smoke requires CUDA.")
def test_cuda_gsplat360_one_step_smoke(tmp_path: Path) -> None:
    if _optional_gsplat360() is None:
        pytest.skip("gsplat360 CUDA extension is unavailable")
    config = copy.deepcopy(default_config())
    config["stage2"]["enabled"] = True
    config["image"] = {"height": 16, "width": 32, "head_height": 16, "head_width": 32}
    config["dataset"].update({"synthetic": True, "views_per_sample": 2, "max_train_samples": 1, "max_val_samples": 1})
    config["head"].update({"channels": [8, 16, 24, 32], "mlp_hidden_dim": 16})
    config["train"].update(
        {
            "feature_device": "cuda:0",
            "train_device": "cuda:0",
            "max_steps": 1,
            "val_interval": 1,
            "save_interval": 1,
            "output_dir": str(tmp_path),
            "amp": False,
        }
    )
    result = train_spherical_selfi_gaussian_head(config)
    assert result["step"] == 1
    assert Path(result["checkpoint"]).is_file()
    assert torch.isfinite(torch.tensor(result["metrics"]["loss"]))
