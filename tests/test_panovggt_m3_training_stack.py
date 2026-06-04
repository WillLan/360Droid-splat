from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from frontend.pano_vggt.matching_dataset import (
    SyntheticOmni360TrainingDataset,
    _load_semantic,
    sky_mask_from_semantic,
    validate_pose_rotation,
)
from frontend.pano_vggt.matching_head import PanoVGGTMatchingSkyHead
from frontend.pano_vggt.matching_losses import (
    PanoVGGTMatchingLossWeights,
    PanoVGGTMatchingSkyLoss,
    sky_bce_dice_loss,
    spherical_match_regression_loss,
    symmetric_info_nce_loss,
)
from frontend.pano_vggt.spherical_correspondence import (
    generate_gt_spherical_correspondences,
    spherical_tangent_residual,
)
from frontend.pano_vggt.train_matching import (
    ExternalPanoVGGTFeatureExtractor,
    FrozenSyntheticFeatureExtractor,
    load_head_checkpoint,
    load_matching_train_config,
    save_combined_head_bundle,
    save_matching_head_checkpoint,
    save_sky_head_checkpoint,
    train_matching,
)


def _config(tmp_path: Path, *, mode: str, variant: str = "complete") -> dict:
    cfg = load_matching_train_config(None)
    cfg["Training"].update(
        {
            "mode": mode,
            "steps": 1,
            "batch_size": 1,
            "frames_per_sample": 3,
            "num_workers": 0,
            "output_dir": str(tmp_path),
            "save_interval": 1,
            "log_interval": 100,
        }
    )
    cfg["Model"].update({"use_synthetic_features": True, "feature_dim": 8, "feature_stride": 4})
    cfg["Heads"].update({"descriptor_dim": 24, "hidden_dim": 8, "num_conv_blocks": 1})
    cfg["Dataset"].update(
        {
            "synthetic": True,
            "synthetic_variant": variant,
            "synthetic_length": 2,
            "height": 16,
            "width": 32,
            "class_map": {"sky_ids": [1], "classes": {"sky": 1}},
        }
    )
    cfg["Pairs"].update({"samples_per_edge": 32, "min_baseline_deg": 0.0, "max_baseline_deg": 60.0})
    cfg["WeightsAndBiases"]["enabled"] = False
    cfg["Validation"].update({"enabled": True, "max_batches": 1, "num_workers": 0})
    cfg["Visualization"].update({"enabled": True, "interval": 1, "max_matches": 8})
    return cfg


def _make_correspondence(feature_hw: tuple[int, int] = (4, 6)):
    depths = torch.full((2, 1, 8, 12), 2.0)
    poses = torch.eye(4).view(1, 4, 4).repeat(2, 1, 1)
    return generate_gt_spherical_correspondences(
        depths,
        poses,
        torch.tensor([[0, 1]]),
        feature_hw=feature_hw,
        image_hw=(8, 12),
        min_baseline_deg=0.0,
        samples_per_edge=feature_hw[0] * feature_hw[1],
    )


def test_synthetic_complete_sample_has_all_training_fields():
    sample = SyntheticOmni360TrainingDataset(variant="complete", mode="matching_only")[0]
    assert sample["images"].shape[:2] == (3, 3)
    assert sample["depths"].shape == (3, 1, 32, 64)
    assert sample["poses_c2w"].shape == (3, 4, 4)
    assert sample["sky_mask"].shape == (3, 1, 32, 64)
    assert sample["has_pose"] is True
    assert sample["has_sky"] is True


def test_synthetic_mode_validation_for_missing_pose_and_sky():
    sky_sample = SyntheticOmni360TrainingDataset(variant="no_pose", mode="sky_only")[0]
    assert sky_sample["poses_c2w"] is None
    assert sky_sample["has_sky"] is True
    with pytest.raises(ValueError, match="requires RGB, depth, and pose"):
        _ = SyntheticOmni360TrainingDataset(variant="no_pose", mode="matching_only")[0]
    with pytest.raises(ValueError, match="requires semantic sky"):
        _ = SyntheticOmni360TrainingDataset(variant="no_sky", mode="sky_only")[0]


def test_sky_class_map_id_name_and_color_paths_work():
    labels = torch.tensor([[0, 2], [1, 2]], dtype=torch.long)
    mask_from_name = sky_mask_from_semantic(labels, None, {"sky_names": ["Sky"], "classes": {"Sky": 2}})
    assert mask_from_name.shape == (1, 2, 2)
    assert mask_from_name.sum() == 2
    rgb = torch.zeros(3, 2, 2)
    rgb[:, 0, 1] = torch.tensor([0.1, 0.4, 0.9])
    mask_from_color = sky_mask_from_semantic(None, rgb, {"sky_colors": [[25.5, 102.0, 229.5]]})
    assert bool(mask_from_color[0, 0, 1])


def test_omni360_rgba_alpha_semantic_labels_are_supported(tmp_path: Path):
    arr = np.zeros((3, 4, 4), dtype=np.uint8)
    arr[..., :3] = np.array([10, 80, 200], dtype=np.uint8)
    arr[..., 3] = np.array(
        [
            [0, 3, 3, 0],
            [1, 1, 3, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "semantic.png"
    Image.fromarray(arr, mode="RGBA").save(path)
    labels, rgb = _load_semantic(path, None)
    assert rgb is None
    assert labels is not None
    mask = sky_mask_from_semantic(labels, None, {"sky_ids": [3]})
    assert int(mask.sum()) == 3


def test_pose_validation_catches_invalid_rotation():
    pose = torch.eye(4).view(1, 4, 4)
    pose[0, 0, 0] = 2.0
    with pytest.raises(ValueError, match="invalid rotation"):
        validate_pose_rotation(pose)


def test_matching_and_sky_head_shapes_ranges_and_descriptor_norms():
    feature = torch.rand(2, 3, 8, 5, 7)
    wrapper = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    out = wrapper(feature)
    assert out["dense_descriptors"].shape == (2, 3, 24, 5, 7)
    assert out["match_confidence"].shape == (2, 3, 1, 5, 7)
    assert out["sky_logits"].shape == (2, 3, 1, 5, 7)
    assert out["sky_prob"].shape == (2, 3, 1, 5, 7)
    assert torch.allclose(out["dense_descriptors"].norm(dim=2), torch.ones(2, 3, 5, 7), atol=1.0e-5)
    assert out["match_confidence"].min() >= 0.0 and out["match_confidence"].max() <= 1.0
    assert out["sky_prob"].min() >= 0.0 and out["sky_prob"].max() <= 1.0


def test_external_feature_hook_accepts_5d_channel_first_and_last_grids():
    extractor = object.__new__(ExternalPanoVGGTFeatureExtractor)
    extractor._input_hw = (56, 84)
    extractor._patch_size = 14
    channel_first = torch.rand(1, 2, 8, 4, 6)
    assert extractor._tokens_to_feature(channel_first).shape == (1, 2, 8, 4, 6)
    channel_last = torch.rand(1, 2, 4, 6, 8)
    assert extractor._tokens_to_feature(channel_last).shape == (1, 2, 8, 4, 6)


def test_info_nce_prefers_correct_positives_over_shuffled_positives():
    corr = _make_correspondence((4, 6))
    desc = torch.zeros(1, 2, 24, 4, 6)
    for idx in range(24):
        y = idx // 6
        x = idx % 6
        desc[0, :, idx, y, x] = 1.0
    desc = torch.nn.functional.normalize(desc, dim=2)
    good, _ = symmetric_info_nce_loss(desc, corr)
    bad_corr = replace(corr, tgt_uv=torch.roll(corr.tgt_uv, shifts=3, dims=1))
    bad, _ = symmetric_info_nce_loss(desc, bad_corr)
    assert good < bad


def test_spherical_regression_confidence_and_sky_losses_are_finite():
    corr = _make_correspondence((4, 6))
    desc = torch.rand(1, 2, 24, 4, 6)
    desc = torch.nn.functional.normalize(desc, dim=2)
    sph, sph_stats = spherical_match_regression_loss(desc, corr, image_hw=(8, 12), search_radius=1)
    sky_logits = torch.randn(1, 2, 1, 4, 6)
    sky_gt = torch.zeros(1, 2, 1, 8, 12, dtype=torch.bool)
    sky_gt[..., :2, :] = True
    sky, sky_stats = sky_bce_dice_loss(sky_logits, sky_gt)
    assert torch.isfinite(sph)
    assert torch.isfinite(sky)
    assert sph_stats["spherical_n"] > 0
    assert 0.0 <= float(sky_stats["sky_iou"]) <= 1.0
    assert 0.0 <= float(sky_stats["sky_pixel_acc"]) <= 1.0


def test_total_losses_backward_only_updates_selected_heads():
    sample = SyntheticOmni360TrainingDataset(variant="complete", mode="matching_only", height=16, width=32)[0]
    images = sample["images"].unsqueeze(0)
    extractor = FrozenSyntheticFeatureExtractor(feature_dim=8, feature_stride=4)
    wrapper = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    features = extractor(images)
    out = wrapper(features)
    corr = generate_gt_spherical_correspondences(
        sample["depths"],
        sample["poses_c2w"],
        sample["pair_indices"],
        feature_hw=tuple(features.shape[-2:]),
        image_hw=tuple(images.shape[-2:]),
        samples_per_edge=32,
        min_baseline_deg=0.0,
    )
    loss_fn = PanoVGGTMatchingSkyLoss(PanoVGGTMatchingLossWeights())
    loss, _ = loss_fn.matching_only(out, corr, image_hw=tuple(images.shape[-2:]))
    loss.backward()
    assert all(param.grad is None for param in extractor.parameters())
    assert any(param.grad is not None for param in wrapper.matching_head.parameters())
    assert all(param.grad is None for param in wrapper.sky_head.parameters())

    sky_wrapper = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    sky_features = extractor(images)
    sky_out = sky_wrapper(sky_features)
    sky_loss, _ = loss_fn.sky_only(sky_out, {"sky_mask": sample["sky_mask"].unsqueeze(0)})
    sky_loss.backward()
    assert any(param.grad is not None for param in sky_wrapper.sky_head.parameters())


def test_training_one_step_modes_and_missing_supervision_errors(tmp_path: Path):
    sky_result = train_matching(_config(tmp_path / "sky", mode="sky_only", variant="no_pose"))
    assert Path(sky_result["checkpoint"]).name == "sky_head.pt"
    assert any((tmp_path / "sky" / "visualizations").glob("*sky.png"))
    assert "val/sky_iou" in sky_result["last_metrics"]
    matching_result = train_matching(_config(tmp_path / "matching", mode="matching_only", variant="complete"))
    assert Path(matching_result["checkpoint"]).name == "matching_head.pt"
    assert any((tmp_path / "matching" / "visualizations").glob("*matching.png"))
    assert any((tmp_path / "matching" / "visualizations").glob("*match_confidence.png"))
    assert "val/precision_at_0_5deg" in matching_result["last_metrics"]
    with pytest.raises(ValueError, match="requires RGB, depth, and pose"):
        train_matching(_config(tmp_path / "bad", mode="matching_only", variant="no_pose"))


def test_head_checkpoints_save_reload_and_combined_bundle(tmp_path: Path):
    wrapper = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    feature = torch.rand(1, 2, 8, 4, 6)
    wrapper.eval()
    with torch.no_grad():
        expected = wrapper(feature)
    cfg = _config(tmp_path, mode="matching_only")
    sky_path = tmp_path / "sky_head.pt"
    match_path = tmp_path / "matching_head.pt"
    bundle_path = tmp_path / "bundle.pt"
    save_sky_head_checkpoint(sky_path, wrapper=wrapper, config=cfg, global_step=3, metrics={"loss": 1.0})
    save_matching_head_checkpoint(match_path, wrapper=wrapper, config=cfg, global_step=3, metrics={"loss": 1.0})
    save_combined_head_bundle(bundle_path, wrapper=wrapper, config=cfg)

    loaded = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    load_head_checkpoint(sky_path, loaded)
    load_head_checkpoint(match_path, loaded)
    loaded.eval()
    with torch.no_grad():
        actual = loaded(feature)
    assert torch.allclose(actual["dense_descriptors"], expected["dense_descriptors"], atol=1.0e-6)
    assert torch.allclose(actual["sky_prob"], expected["sky_prob"], atol=1.0e-6)
    bundled = PanoVGGTMatchingSkyHead(8, descriptor_dim=24, hidden_dim=8, num_conv_blocks=1)
    payload = load_head_checkpoint(bundle_path, bundled)
    assert payload["format"] == "panovggt_m3_sphere_matching_sky_bundle_v1"


def test_spherical_regression_loss_uses_tangent_residual_not_pixel_delta():
    source = inspect.getsource(spherical_match_regression_loss)
    assert "spherical_tangent_residual" in source
    assert "seam_aware_delta" not in source
    target = torch.tensor([[0.0, 0.0, 1.0]])
    residual = spherical_tangent_residual(target, target)
    assert residual.abs().max() < 1.0e-5
