from pathlib import Path

from system.pano_droid_gs_slam import load_config


ROOT = Path(__file__).parents[1]


def _load(name: str) -> dict:
    return load_config(ROOT / "configs" / name)


def test_stride4_experiment_keeps_abd6213_single_stage_ba() -> None:
    config = _load("boundary_globalba25_stride4_margin0_abd6213_norefiner.yaml")
    local_ba = config["SphericalSelfiRuntime"]["local_ba"]
    graph = config["SphericalSelfiGlobalBackend"]["global_graph"]
    map_optimization = config["SphericalSelfiGlobalBackend"]["map_optimization"]

    assert config["Dataset"]["frame_stride"] == 4
    assert config["Dataset"]["end"] == 100
    assert local_ba["iterations"] == 5
    assert "outlier_refinement" not in local_ba
    assert local_ba["matching"]["type"] == "adapter"
    assert local_ba["matching"]["num_queries"] == 1024
    assert graph["min_match_margin"] == 0.0
    assert graph["umeyama_irls_iterations"] == 3
    assert map_optimization["pose_refine_enable"] is False
    assert map_optimization["pose_lr"] == 0.0
    assert "VoxelAnchorRefiner" not in config


def test_pose_refine_experiment_changes_only_requested_controls() -> None:
    baseline = _load("boundary_globalba100_margin0_abd6213_base.yaml")
    config = _load(
        "boundary_globalba100_localba8_pose2e4_umeyama5_abd6213_norefiner.yaml"
    )

    assert config["Dataset"]["frame_stride"] == 1
    assert config["SphericalSelfiRuntime"]["local_ba"]["iterations"] == 8
    assert (
        config["SphericalSelfiGlobalBackend"]["global_graph"][
            "umeyama_irls_iterations"
        ]
        == 5
    )
    map_optimization = config["SphericalSelfiGlobalBackend"]["map_optimization"]
    assert map_optimization["pose_refine_enable"] is True
    assert map_optimization["pose_lr"] == 2.0e-4
    assert "outlier_refinement" not in config["SphericalSelfiRuntime"]["local_ba"]
    assert "VoxelAnchorRefiner" not in config

    ignored_paths = {
        ("SphericalSelfiRuntime", "local_ba", "iterations"),
        ("SphericalSelfiGlobalBackend", "global_graph", "umeyama_irls_iterations"),
        ("SphericalSelfiGlobalBackend", "map_optimization", "pose_refine_enable"),
        ("SphericalSelfiGlobalBackend", "map_optimization", "pose_lr"),
        ("WeightsAndBiases", "run_name"),
        ("WeightsAndBiases", "tags"),
        ("Results", "save_dir"),
    }

    def flatten(value, prefix=()):
        if isinstance(value, dict):
            for key, child in value.items():
                yield from flatten(child, prefix + (key,))
        else:
            yield prefix, value

    baseline_values = dict(flatten(baseline))
    config_values = dict(flatten(config))
    changed = {
        path
        for path in baseline_values.keys() | config_values.keys()
        if baseline_values.get(path) != config_values.get(path)
    }
    assert changed == ignored_paths
