from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

import tools.formal_experiments as formal_experiments
from system.pano_droid_gs_slam import load_config
from tools.formal_experiments import (
    RunSpec,
    _assert_formal_mainline,
    _assert_dataset_policy,
    _deep_merge_config,
    _expand_runs,
    _prepare_torch_home,
    _sha256,
    _verify_dataset_run,
    _wait_until_resources_ready,
    validate_run,
)
from tools.formal_phase_supervisor import phase_status
from tools.formal_cleanup_monitor import completed_runs


def _campaign() -> dict:
    path = Path(__file__).parents[1] / "configs/formal/panogsslam_formal_campaign_v2.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _v3_campaign(name: str) -> dict:
    path = Path(__file__).parents[1] / f"configs/formal/{name}"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_formal_campaign_expands_to_balanced_34_run_queue() -> None:
    runs = _expand_runs(_campaign())

    assert len(runs) == 34
    assert sum(run.dataset == "ob3d" for run in runs) == 24
    assert sum(run.dataset == "360vo" for run in runs) == 10
    assert {run.worker for run in runs} == {0, 1}
    loads = [sum(run.frames for run in runs if run.worker == worker) for worker in (0, 1)]
    assert loads == [3700, 3700]


def test_v3_campaigns_are_balanced_independent_phases() -> None:
    ob3d = _expand_runs(_v3_campaign("panogsslam_formal_ob3d_v3.yaml"))
    vo = _expand_runs(_v3_campaign("panogsslam_formal_360vo200_v3.yaml"))

    assert len(ob3d) == 24
    assert len(vo) == 10
    assert [sum(run.frames for run in ob3d if run.worker == worker) for worker in (0, 1)] == [1200, 1200]
    assert [sum(run.frames for run in vo if run.worker == worker) for worker in (0, 1)] == [1000, 1000]
    assert all(run.frames == 200 and run.dataset == "360vo" for run in vo)


def test_rar_pano_campaign_uses_every_frame_and_ob3d_geometry_policy() -> None:
    root = Path(__file__).parents[1]
    campaign = _v3_campaign("panogsslam_formal_rar_pano_v10.yaml")
    runs = _expand_runs(campaign)
    expected_frames = {
        "I_alley": 145,
        "I_avenue": 272,
        "I_bridge": 147,
        "I_bypath": 86,
        "I_garden": 210,
        "O_car": 188,
        "O_lion": 253,
        "O_statuary": 270,
        "O_stone": 171,
    }

    assert len(runs) == 9
    assert {run.scene: run.frames for run in runs} == expected_frames
    assert all(run.dataset_type == "rar_pano" for run in runs)
    assert all(run.split == "Full" for run in runs)
    assert [sum(run.frames for run in runs if run.worker == worker) for worker in (0, 1)] == [
        903,
        839,
    ]

    base = load_config(root / campaign["base_config"])
    resolved = _deep_merge_config(
        copy.deepcopy(base),
        runs[0].config_overrides,
    )
    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, runs[0])
    assert resolved["SphericalSelfiRuntime"]["sky"]["enabled"] is False
    assert resolved["Mapping"]["sky_mask_enable"] is False
    assert resolved["SkyBox"]["enabled"] is False
    assert resolved["SkySphere"]["enabled"] is False
    pfgs = resolved["SphericalSelfiGlobalBackend"]["map_optimization"][
        "pfgs360"
    ]
    assert pfgs["atomic_refined_anchor_replacement"] is True
    assert pfgs["append_only_refined_anchors"] is True
    assert pfgs["growth_hash_dedup_enabled"] is False
    assert pfgs["anchor_footprint"] == {
        "enabled": True,
        "sigma": 2.0,
        "min_radius_pixels": 1.0,
        "max_radius_pixels": 8.0,
        "min_pixels": 1,
        "min_coverage": 0.05,
    }
    assert (
        resolved["Dataset"]["reference_pose_source"]
        == "opensfm_reconstruction_pseudo_gt"
    )
    assert resolved["Dataset"]["reference_pose_usage"] == "evaluation_only"


def test_rar_pano_voxel004_geometry_sky_campaign_is_three_worker_safe_queue() -> None:
    root = Path(__file__).parents[1]
    campaign = _v3_campaign(
        "panogsslam_formal_rar_pano_v11_voxel004_skygeom.yaml"
    )
    runs = _expand_runs(campaign)
    assert len(runs) == 9
    assert {run.worker for run in runs} == {0, 1, 2}
    assert [
        sum(run.frames for run in runs if run.worker == worker)
        for worker in (0, 1, 2)
    ] == [590, 603, 549]

    base = load_config(root / campaign["base_config"])
    resolved = _deep_merge_config(
        copy.deepcopy(base),
        runs[0].config_overrides,
    )
    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, runs[0])
    sky = resolved["SphericalSelfiRuntime"]["sky"]
    assert sky["enabled"] is True
    assert sky["required"] is True
    assert sky["threshold"] == 0.6
    assert sky["geometry_only"] is True
    assert resolved["Mapping"]["sky_mask_enable"] is False
    assert resolved["SkyBox"]["enabled"] is False
    assert resolved["SkySphere"]["enabled"] is False
    assert resolved["VoxelAnchorRefiner"]["voxel_sizes"] == [
        0.04,
        0.08,
        0.16,
        0.32,
    ]
    assert resolved["VoxelAnchorRefiner"]["allow_voxel_size_override"] is False
    assert resolved["SphericalSelfiGlobalBackend"]["voxel_fusion"][
        "voxel_sizes"
    ] == [0.04, 0.08, 0.16, 0.32]
    assert resolved["Runtime"]["cpu_threading"] == {
        "enabled": True,
        "intraop_threads": 2,
        "interop_threads": 1,
        "native_threads": 2,
        "opencv_threads": 1,
    }
    assert campaign["resource_limits"]["wait_for_resources"] is True
    assert campaign["resource_limits"]["max_attempts"] == 4


def test_rar_pano_depthbins_voxel002_campaign_explicitly_overrides_checkpoint() -> None:
    root = Path(__file__).parents[1]
    campaign = _v3_campaign(
        "panogsslam_formal_rar_pano_v12_depthbins_voxel002_skygeom.yaml"
    )
    runs = _expand_runs(campaign)
    assert len(runs) == 9
    assert {run.worker for run in runs} == {0, 1, 2}
    assert [
        sum(run.frames for run in runs if run.worker == worker)
        for worker in (0, 1, 2)
    ] == [590, 603, 549]

    base = load_config(root / campaign["base_config"])
    resolved = _deep_merge_config(
        copy.deepcopy(base),
        runs[0].config_overrides,
    )
    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, runs[0])
    refiner = resolved["VoxelAnchorRefiner"]
    assert refiner["depth_boundaries"] == [2.5, 5.0, 20.0]
    assert refiner["voxel_sizes"] == [0.02, 0.04, 0.16, 0.32]
    assert refiner["allow_depth_boundary_override"] is True
    assert refiner["allow_voxel_size_override"] is True
    assert resolved["SphericalSelfiGlobalBackend"]["voxel_fusion"][
        "voxel_sizes"
    ] == [0.02, 0.04, 0.16, 0.32]
    assert resolved["SphericalSelfiRuntime"]["sky"]["geometry_only"] is True
    assert resolved["Mapping"]["sky_mask_enable"] is False
    assert resolved["SkyBox"]["enabled"] is False
    assert resolved["SkySphere"]["enabled"] is False


def test_rar_pano_recent6_owner_campaign_is_four_worker_default() -> None:
    root = Path(__file__).parents[1]
    campaign = _v3_campaign(
        "panogsslam_formal_rar_pano_v13_recent6owners.yaml"
    )
    runs = _expand_runs(campaign)
    assert len(runs) == 9
    assert {run.worker for run in runs} == {0, 1, 2, 3}

    base = load_config(root / campaign["base_config"])
    resolved = _deep_merge_config(
        copy.deepcopy(base),
        runs[0].config_overrides,
    )
    _assert_formal_mainline(resolved, seed=123)
    _assert_dataset_policy(resolved, runs[0])
    optimize = resolved["SphericalSelfiGlobalBackend"][
        "map_optimization"
    ]
    assert optimize["recent_window_count"] == 3
    assert optimize["camera_steps"] == 50
    assert optimize["joint_steps"] == 200
    assert optimize["gaussian_owner_window_count"] == 6
    assert optimize["pfgs360"]["frame_scope"] == "recent_chunks"
    assert (
        optimize["pfgs360"]["gaussian_update_scope"]
        == "recent_owner_chunks"
    )
    assert resolved["SphericalSelfiRuntime"]["pager_depth"]["enabled"] is True
    assert (
        resolved["SphericalSelfiRuntime"]["local_ba"]["matching"]["type"]
        == "superpoint_sphereglue"
    )
    assert (
        resolved["SphericalSelfiGlobalBackend"][
            "rendered_overlap_alignment"
        ]["mode"]
        == "two_frame_global_map_full_sim3"
    )


def test_rar_pano_official_chunkwise_campaign_passes_formal_guards() -> None:
    root = Path(__file__).parents[1]
    campaign = _v3_campaign(
        "panogsslam_formal_rar_pano_v14_pfgs360_official.yaml"
    )
    runs = _expand_runs(campaign)

    assert len(runs) == 9
    assert {run.worker for run in runs} == {0, 1, 2, 3}
    base = load_config(root / campaign["base_config"])
    for run in runs:
        resolved = _deep_merge_config(
            copy.deepcopy(base),
            run.config_overrides,
        )
        _assert_formal_mainline(resolved, seed=123)
        _assert_dataset_policy(resolved, run)

    optimization = resolved["SphericalSelfiGlobalBackend"]["map_optimization"]
    assert optimization["strategy"] == "pfgs360_official_chunkwise"
    assert optimization["initial_steps"] == 1000
    assert optimization["camera_steps"] == 500
    assert optimization["joint_steps"] == 500
    assert optimization["final_finetune_steps"] == 10000
    assert optimization["pfgs360"]["growth_hash_dedup_enabled"] is True


def test_resource_wait_requires_stable_swap_samples_before_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def snapshot(pswpin: int) -> dict[str, int]:
        return {
            "available_memory": 200 * 1024**3,
            "free_disk": 200 * 1024**3,
            "pswpin": pswpin,
            "pswpout": 0,
            "cpu_count": 100,
            "load_1m_milli": 1000,
        }

    values = iter([snapshot(10), snapshot(11), snapshot(11), snapshot(11)])
    monkeypatch.setattr(
        formal_experiments,
        "_read_resource_snapshot",
        lambda formal_root: next(values),
    )
    monkeypatch.setattr(
        formal_experiments,
        "_gpu_processes",
        lambda gpu: [],
    )
    monkeypatch.setattr(formal_experiments.time, "sleep", lambda seconds: None)
    formal_root = tmp_path / "formal"
    run_root = formal_root / "run"
    formal_root.mkdir()

    result = _wait_until_resources_ready(
        formal_root,
        run_root,
        gpu=3,
        min_memory=80 * 1024**3,
        min_disk=50 * 1024**3,
        max_cpu_fraction=0.8,
        poll_sec=30,
        stable_samples=2,
        wait=True,
    )

    assert result is not None
    assert result["pswpin"] == 11
    assert not (run_root / "paused_resource_guard.json").exists()


def test_rar_pano_dataset_verifier_requires_exact_images_and_opensfm_shots(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "I_alley" / "images"
    image_dir.mkdir(parents=True)
    for name in ("001.png", "002.png"):
        (image_dir / name).write_bytes(b"png")
    (image_dir / "comparison_001.png").write_bytes(b"not a frame")
    (tmp_path / "I_alley" / "reconstruction.json").write_text(
        json.dumps(
            [
                {
                    "shots": {
                        name: {
                            "rotation": [0.0, 0.0, 0.0],
                            "translation": [0.0, 0.0, 0.0],
                        }
                        for name in ("001.png", "002.png")
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    run = RunSpec(
        run_id="rar_pano__i_alley__full",
        dataset="rar_pano",
        dataset_type="rar_pano",
        root=str(tmp_path),
        scene="I_alley",
        split="Full",
        frames=2,
        config_overrides={},
        worker=0,
    )

    _verify_dataset_run(run)

    mismatched = RunSpec(**{**run.__dict__, "frames": 1})
    with pytest.raises(ValueError, match="all-frame run requires exactly 1"):
        _verify_dataset_run(mismatched)


def test_formal_base_config_locks_the_confirmed_mainline() -> None:
    root = Path(__file__).parents[1]
    config = load_config(
        root / "configs/formal/panogsslam_pager_globalmap_refinedanchor_50_200_v1.yaml"
    )

    _assert_formal_mainline(config, seed=123)
    assert "/artifacts/checkpoints/panogsslam_formal_mainline_v1/" in config["panovggt"]["checkpoint"]
    assert "/outputs/" not in config["adapter_checkpoint"]["path"]
    assert config["Results"]["save_final_checkpoint"] is True
    assert config["Results"]["final_image_metrics"] == "pfgs360_official"


def test_formal_v2_applies_dataset_specific_sky_and_voxel_policies() -> None:
    root = Path(__file__).parents[1]
    campaign = _campaign()
    base = load_config(
        root / "configs/formal/panogsslam_pager_globalmap_refinedanchor_50_200_v2.yaml"
    )
    runs = _expand_runs(campaign)

    for dataset in ("ob3d", "360vo"):
        run = next(value for value in runs if value.dataset == dataset)
        resolved = _deep_merge_config(copy.deepcopy(base), run.config_overrides)
        _assert_formal_mainline(resolved, seed=123)
        _assert_dataset_policy(resolved, run)

    ob3d = next(value for value in runs if value.dataset == "ob3d")
    ob3d_config = _deep_merge_config(copy.deepcopy(base), ob3d.config_overrides)
    assert ob3d_config["SphericalSelfiRuntime"]["sky"]["enabled"] is False
    assert ob3d_config["SkyBox"]["enabled"] is False
    assert ob3d_config["VoxelAnchorRefiner"]["voxel_sizes"] == [
        0.02,
        0.04,
        0.08,
        0.16,
    ]
    assert ob3d_config["VoxelAnchorRefiner"]["allow_voxel_size_override"] is True

    vo = next(value for value in runs if value.dataset == "360vo")
    vo_config = _deep_merge_config(copy.deepcopy(base), vo.config_overrides)
    assert vo_config["SphericalSelfiRuntime"]["sky"]["threshold"] == 0.6
    assert vo_config["SkyBox"]["enabled"] is True
    assert vo_config["VoxelAnchorRefiner"]["voxel_sizes"] == [
        0.04,
        0.08,
        0.16,
        0.32,
    ]
    assert vo_config["VoxelAnchorRefiner"]["allow_voxel_size_override"] is False


def test_writable_torch_home_copies_and_verifies_lpips_backbone(tmp_path: Path) -> None:
    immutable = tmp_path / "immutable/alexnet-owt-7be5be79.pth"
    immutable.parent.mkdir()
    immutable.write_bytes(b"offline alexnet weights")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "role": "lpips_alexnet_backbone",
                        "destination": str(immutable),
                        "size_bytes": immutable.stat().st_size,
                        "sha256": _sha256(immutable),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    formal_root = tmp_path / "formal"
    formal_root.mkdir()

    torch_home = _prepare_torch_home(
        {"torch_home": "runtime_cache/torch"},
        formal_root=formal_root,
        weight_manifest=manifest,
    )

    assert torch_home == (formal_root / "runtime_cache/torch").resolve()
    copied = torch_home / "hub/checkpoints" / immutable.name
    assert copied.read_bytes() == immutable.read_bytes()
    assert _sha256(copied) == _sha256(immutable)
    assert (formal_root / "runtime_cache/manifest.json").is_file()


def test_phase_status_enforces_hard_count_and_failed_runs(tmp_path: Path) -> None:
    root = tmp_path / "phase"
    runs = [{"run_id": f"run_{index}"} for index in range(3)]
    root.mkdir()
    (root / "campaign.json").write_text(
        json.dumps({"expected_run_count": 3, "runs": runs}), encoding="utf-8"
    )
    (root / "runs/run_0").mkdir(parents=True)
    (root / "runs/run_0/complete.marker").write_text("{}", encoding="utf-8")
    (root / "runs/run_1").mkdir(parents=True)
    (root / "runs/run_1/failed.json").write_text("{}", encoding="utf-8")

    assert phase_status(root, expected_run_count=3) == {
        "complete": 1,
        "failed": ["run_1"],
        "pending": 1,
    }


def test_cleanup_counter_ignores_smoke_markers(tmp_path: Path) -> None:
    (tmp_path / "ob3d/runs/a").mkdir(parents=True)
    (tmp_path / "ob3d/runs/a/complete.marker").write_text("ok", encoding="utf-8")
    (tmp_path / "smoke/ob3d_metric_4frame").mkdir(parents=True)
    (tmp_path / "smoke/ob3d_metric_4frame/complete.marker").write_text(
        "ok", encoding="utf-8"
    )

    assert completed_runs(tmp_path) == 1


def test_formal_run_validator_requires_paper_artifact_contract(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt_001"
    trajectory = attempt / "final_all_frames/trajectory"
    renders = attempt / "final_all_frames/render_rgb"
    checkpoint = attempt / "checkpoints/final_gaussian_map.pt"
    trajectory.mkdir(parents=True)
    renders.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    frame_count = 12
    poses = [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]] * frame_count
    for index in range(frame_count):
        (renders / f"frame_{index:06d}.png").write_bytes(b"png")
    for filename in ("predicted_c2w.json", "gt_c2w.json", "sim3_aligned_predicted_c2w.json"):
        (trajectory / filename).write_text(json.dumps({"poses": poses}), encoding="utf-8")
    (trajectory / "trajectory_sim3.png").write_bytes(b"png")
    (trajectory / "metrics.json").write_text("{}", encoding="utf-8")
    metrics = {
        "render_count": frame_count,
        "ate_count": frame_count,
        "mean_psnr": 20.0,
        "mean_ssim": 0.8,
        "mean_lpips": 0.2,
        **{key: 0.1 for key in (
            "pfgs360_ate", "sim3_ate_rmse", "se3_ate_rmse",
            "rpe_delta_1_translation_rmse", "rpe_delta_1_rotation_mean_deg",
            "rpe_delta_3_translation_rmse", "rpe_delta_10_translation_rmse",
            "scale_drift_percent", "path_length_scale_ratio",
        )},
    }
    (attempt / "final_all_frames/metrics.json").write_text(
        json.dumps({"metrics": metrics}), encoding="utf-8"
    )
    (attempt / "summary.json").write_text(
        json.dumps({"frames": frame_count}), encoding="utf-8"
    )
    (attempt / "runtime.json").write_text(
        json.dumps({"total_wall_sec": 10.0, "seconds_per_frame": 0.8, "fps": 1.2}),
        encoding="utf-8",
    )
    (attempt / "run_provenance.json").write_text(
        json.dumps({"weights_manifest_sha256": "abc"}), encoding="utf-8"
    )

    result = validate_run(
        attempt,
        expected_frames=frame_count,
        expected_weights_manifest_sha256="abc",
    )

    assert result["valid"] is True
    assert (attempt / "validation.json").is_file()
