"""Generate reproducible 200-step Stage 3 BA ablation configurations.

The generated experiments hold matching, data, Refiner, seed, DDP, and loss
settings constant.  They differ only in dense-depth propagation, scale gauge,
and nonlinear solver so their individual effects can be attributed.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


EXPERIMENTS: dict[str, dict[str, str]] = {
    "e0_ba0_affine_backtracking": {
        "dense_depth_mode": "affine",
        "gauge_mode": "none",
        "solver_mode": "backtracking_gn",
    },
    "e1_ba0_noaffine_backtracking": {
        "dense_depth_mode": "none",
        "gauge_mode": "none",
        "solver_mode": "backtracking_gn",
    },
    "e2_ba0_affine_gauge_backtracking": {
        "dense_depth_mode": "affine",
        "gauge_mode": "initial_baseline",
        "solver_mode": "backtracking_gn",
    },
    "e3_ba0_affine_standard_lm": {
        "dense_depth_mode": "affine",
        "gauge_mode": "none",
        "solver_mode": "standard_lm",
    },
    "e4_ba0_noaffine_gauge_standard_lm": {
        "dense_depth_mode": "none",
        "gauge_mode": "initial_baseline",
        "solver_mode": "standard_lm",
    },
}


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return payload


def generate(base_path: Path, suite_dir: Path) -> dict[str, Any]:
    base = _load(base_path)
    config_dir = suite_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "base_config": str(base_path.resolve()),
        "controlled_variables": [
            "dense_depth_mode",
            "gauge_mode",
            "solver_mode",
        ],
        "experiments": [],
    }
    for name, ba_variant in EXPERIMENTS.items():
        config = copy.deepcopy(base)
        config["experiment_name"] = name
        config["ba"]["outer_schedule"] = [True, False, False]
        config["ba"].update(ba_variant)
        config["loss"]["dssim"] = 0.0
        config["train"].update(
            {
                "max_steps": 200,
                "diagnostics_interval": 200,
                "val_interval": 200,
                "save_interval": 200,
                "output_dir": str(suite_dir / name),
                "resume": None,
            }
        )
        config["WeightsAndBiases"].update(
            {
                "enabled": True,
                "mode": "online",
                "run_name": name,
                "tags": list(config["WeightsAndBiases"].get("tags", []))
                + ["ba-ablation-200", name],
            }
        )
        config["Visualization"]["enabled"] = True
        output_path = config_dir / f"{name}.yaml"
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(config, handle, sort_keys=False)
        manifest["experiments"].append(
            {
                "name": name,
                "config": str(output_path),
                "output_dir": config["train"]["output_dir"],
                "ba": ba_variant,
            }
        )
    manifest_path = suite_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = generate(args.base, args.suite_dir)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
