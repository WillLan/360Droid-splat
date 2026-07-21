from __future__ import annotations

from pathlib import Path

import pytest
import torch

from frontend.spherical_selfi.pager_depth import (
    PaGeRDepthAlignmentError,
    PaGeRDepthConfig,
    PaGeRDepthProvider,
    align_pager_depth_to_panovggt,
)
from system.pano_droid_gs_slam import _SLAM_CORE_VISUAL_WANDB_KEYS, load_config


def test_pager_depth_config_is_strict_and_disabled_by_default() -> None:
    disabled = PaGeRDepthConfig.from_mapping(None)
    assert disabled.enabled is False
    assert disabled.checkpoint == "prs-eth/PaGeR"
    assert disabled.alignment_mode == "per_frame_panovggt_scale"

    with pytest.raises(ValueError, match="repo_path"):
        PaGeRDepthConfig.from_mapping({"enabled": True})
    with pytest.raises(ValueError, match="apply_head_depth_residual=false"):
        PaGeRDepthConfig.from_mapping(
            {
                "enabled": True,
                "repo_path": "/unused/pager",
                "apply_head_depth_residual": True,
            }
        )
    with pytest.raises(ValueError, match="only 'error'"):
        PaGeRDepthConfig.from_mapping(
            {
                "enabled": True,
                "repo_path": "/unused/pager",
                "failure_policy": "fallback",
            }
        )


def test_per_frame_log_weighted_median_recovers_scale_with_outliers_and_sky() -> None:
    pager = torch.linspace(0.5, 4.0, steps=64).reshape(1, 2, 1, 4, 8)
    pano = pager * torch.tensor([2.0, 5.0]).view(1, 2, 1, 1, 1)
    pano[:, :, :, 0, :2] *= 100.0
    sky = torch.zeros_like(pager, dtype=torch.bool)
    sky[:, :, :, 0, :2] = True
    aligned, diagnostics = align_pager_depth_to_panovggt(
        pager,
        pano,
        sky_mask=sky,
        frame_ids=torch.tensor([[10, 11]]),
        min_valid_pixels=8,
        min_valid_ratio=0.25,
    )
    expected = pager * torch.tensor([2.0, 5.0]).view(1, 2, 1, 1, 1)
    torch.testing.assert_close(aligned, expected)
    assert [row["frame_id"] for row in diagnostics] == [10, 11]
    assert diagnostics[0]["scale"] == pytest.approx(2.0)
    assert diagnostics[1]["scale"] == pytest.approx(5.0)
    assert max(row["log_mad"] for row in diagnostics) < 1.0e-6


def test_pager_alignment_fails_instead_of_falling_back() -> None:
    pager = torch.ones(1, 1, 1, 4, 8)
    pano = torch.ones_like(pager)
    sky = torch.ones_like(pager, dtype=torch.bool)
    with pytest.raises(PaGeRDepthAlignmentError, match="insufficient"):
        align_pager_depth_to_panovggt(
            pager,
            pano,
            sky_mask=sky,
            frame_ids=torch.tensor([[7]]),
            min_valid_pixels=1,
            min_valid_ratio=0.01,
        )


def test_pager_provider_caches_raw_depth_by_frame_id_across_windows() -> None:
    calls: list[int] = []

    def fake_inference(images: torch.Tensor) -> torch.Tensor:
        calls.append(int(images.shape[0]))
        return images.mean(dim=1, keepdim=True) + 1.0

    config = PaGeRDepthConfig.from_mapping(
        {
            "enabled": True,
            "repo_path": "/unused/pager",
            "micro_batch_size": 1,
            "cache_size": 3,
            "min_valid_pixels": 1,
        }
    )
    provider = PaGeRDepthProvider(
        config,
        device=torch.device("cpu"),
        infer_batch_fn=fake_inference,
    )
    first_images = torch.stack(
        [torch.zeros(3, 4, 8), torch.ones(3, 4, 8)], dim=0
    ).unsqueeze(0)
    first, first_diag = provider.predict(first_images, torch.tensor([[10, 11]]))
    assert calls == [1, 1]
    assert first_diag["cache_hits"] == 0
    assert first_diag["cache_misses"] == 2
    torch.testing.assert_close(first[0, 1], torch.full((1, 4, 8), 2.0))

    second_images = torch.stack(
        [torch.full((3, 4, 8), 9.0), torch.full((3, 4, 8), 2.0)], dim=0
    ).unsqueeze(0)
    second, second_diag = provider.predict(second_images, torch.tensor([[11, 12]]))
    assert calls == [1, 1, 1]
    assert second_diag["cache_hits"] == 1
    assert second_diag["cache_misses"] == 1
    torch.testing.assert_close(second[0, 0], torch.full((1, 4, 8), 2.0))
    torch.testing.assert_close(second[0, 1], torch.full((1, 4, 8), 3.0))
    assert provider.cache_size == 3

    provider.reset()
    assert provider.cache_size == 0


def test_pager_refined_anchor_pointmap_mainline_resolves_required_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root
        / "configs"
        / "spherical_selfi_ob3d_pointmap_sim3_sphereglue_pager_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )
    runtime = config["SphericalSelfiRuntime"]
    pager = runtime["pager_depth"]
    local_ba = runtime["local_ba"]
    global_backend = config["SphericalSelfiGlobalBackend"]

    assert pager["enabled"] is True
    assert pager["output"] == "scale_invariant"
    assert pager["checkpoint"].endswith("/PaGeR-checkpoints/unified")
    assert pager["apply_head_depth_residual"] is False
    assert pager["failure_policy"] == "error"
    assert local_ba["matching"]["type"] == "superpoint_sphereglue"
    assert local_ba["depth_parameterization"] == "fixed"
    assert local_ba["dense_depth_mode"] == "none"
    assert local_ba["defer_dense_depth_affine"] is False
    assert (
        global_backend["rendered_overlap_alignment"]["mode"]
        == "two_frame_pointmap_full_sim3"
    )
    assert (
        global_backend["rendered_overlap_alignment"]["acceptance_policy"]
        == "diagnostics_only"
    )
    assert global_backend["global_graph"]["node_mode"] == "chunk_first_stride"
    optimization = global_backend["map_optimization"]
    assert optimization["strategy"] == "pfgs360_full_50_50"
    assert optimization["camera_steps"] == 50
    assert optimization["joint_steps"] == 50
    assert optimization["pfgs360"]["growth_source"] == "refined_anchor"
    assert optimization["pfgs360"]["bootstrap_source"] == (
        "refined_anchor_all_views"
    )
    assert optimization["pfgs360"]["state_storage_device"] == "map"
    assert global_backend["insertion_dedup"]["radius_voxels"] == pytest.approx(1.0)
    assert config["Runtime"]["cpu_threading"]["intraop_threads"] == 8
    assert config["WeightsAndBiases"]["runtime_log_preset"] == "slam_core_visuals"
    assert config["WeightsAndBiases"]["enabled"] is True
    assert config["Visualization"]["enabled"] is True


def test_pager_global_map_pair_only_changes_alignment_and_run_identity() -> None:
    root = Path(__file__).resolve().parents[1] / "configs"
    pointmap = load_config(
        root
        / "spherical_selfi_ob3d_pointmap_sim3_sphereglue_pager_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )
    global_map = load_config(
        root
        / "spherical_selfi_ob3d_global_map_sim3_sphereglue_pager_ba_100_pfgs360_refined_anchor_50_50.yaml"
    )
    assert pointmap["SphericalSelfiGlobalBackend"]["rendered_overlap_alignment"][
        "mode"
    ] == "two_frame_pointmap_full_sim3"
    assert global_map["SphericalSelfiGlobalBackend"]["rendered_overlap_alignment"][
        "mode"
    ] == "two_frame_global_map_full_sim3"

    import copy

    normalized_pointmap = copy.deepcopy(pointmap)
    normalized_global = copy.deepcopy(global_map)
    normalized_pointmap["SphericalSelfiGlobalBackend"][
        "rendered_overlap_alignment"
    ]["mode"] = "two_frame_global_map_full_sim3"
    for config in (normalized_pointmap, normalized_global):
        for key in ("group", "run_name", "tags"):
            config["WeightsAndBiases"].pop(key, None)
        config["Results"].pop("save_dir", None)
    assert normalized_global == normalized_pointmap


def test_pager_aggregate_metrics_are_allowed_by_core_wandb_preset() -> None:
    expected = {
        "pager_depth/enabled",
        "pager_depth/inference_sec",
        "pager_depth/alignment_sec",
        "pager_depth/cache_hits",
        "pager_depth/cache_misses",
        "pager_depth/cache_hit_ratio",
        "pager_depth/cache_entries",
        "pager_depth/scale_mean",
        "pager_depth/scale_min",
        "pager_depth/scale_max",
        "pager_depth/log_mad_mean",
        "pager_depth/valid_ratio_mean",
    }
    assert expected <= _SLAM_CORE_VISUAL_WANDB_KEYS
