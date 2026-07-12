"""Generate the reproducible full-resolution 200-step Stage 3 BA pilot config."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def generate(
    base_path: Path,
    output_path: Path,
    *,
    output_dir: str,
    run_name: str,
    max_steps: int = 200,
) -> dict[str, Any]:
    config = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {base_path}.")
    config["experiment_name"] = run_name
    config["matching"].update(
        {
            "reliability_keep_fraction": 0.10,
            "distinctiveness_exclusion_deg": 0.0,
            "subpixel_refine_radius": 0,
        }
    )
    config["ba"].update(
        {
            "outer_schedule": [True, False, False],
            "solver_mode": "standard_lm",
            "dense_depth_mode": "none",
            "gauge_mode": "initial_baseline",
            "pose_update_side": "right",
            "pose_dof_mode": "rotation_only",
            "max_pose_update_deg": 0.02,
            "max_translation_update": 0.001,
            "min_initial_median_residual_deg": 0.0,
        }
    )
    config["loss"]["dssim"] = 0.0
    config["train"].update(
        {
            "max_steps": int(max_steps),
            "diagnostics_interval": int(max_steps),
            "val_interval": int(max_steps),
            "save_interval": int(max_steps),
            "output_dir": str(output_dir),
            "resume": None,
        }
    )
    config["WeightsAndBiases"].update(
        {
            "enabled": True,
            "mode": "online",
            "run_name": run_name,
            "tags": list(config["WeightsAndBiases"].get("tags", []))
            + ["stage3-ba-useful-pilot-200", "rotation-only-trust002"],
        }
    )
    config["Visualization"]["enabled"] = True
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()
    generate(
        args.base,
        args.output,
        output_dir=args.output_dir,
        run_name=args.run_name,
        max_steps=max(1, int(args.max_steps)),
    )
    print(args.output)


if __name__ == "__main__":
    main()
