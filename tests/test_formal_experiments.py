from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

from system.pano_droid_gs_slam import load_config
from tools.formal_experiments import (
    _assert_formal_mainline,
    _assert_dataset_policy,
    _deep_merge_config,
    _expand_runs,
    validate_run,
)


def _campaign() -> dict:
    path = Path(__file__).parents[1] / "configs/formal/panogsslam_formal_campaign_v2.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_formal_campaign_expands_to_balanced_34_run_queue() -> None:
    runs = _expand_runs(_campaign())

    assert len(runs) == 34
    assert sum(run.dataset == "ob3d" for run in runs) == 24
    assert sum(run.dataset == "360vo" for run in runs) == 10
    assert {run.worker for run in runs} == {0, 1}
    loads = [sum(run.frames for run in runs if run.worker == worker) for worker in (0, 1)]
    assert loads == [3700, 3700]


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
