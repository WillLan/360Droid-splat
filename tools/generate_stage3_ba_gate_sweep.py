"""Generate fixed-window BA-only factor-gating sweep configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


VARIANTS: dict[str, dict[str, float | None]] = {
    "v0_kkt_all": {"keep": 1.0, "residual": None, "parallax": 0.0},
    "v1_kkt_reliable50": {"keep": 0.50, "residual": None, "parallax": 0.0},
    "v2_kkt_reliable25": {"keep": 0.25, "residual": None, "parallax": 0.0},
    "v3_kkt_reliable10": {"keep": 0.10, "residual": None, "parallax": 0.0},
    "v4_kkt_reliable25_res3": {"keep": 0.25, "residual": 3.0, "parallax": 0.0},
    "v5_kkt_reliable25_res2": {"keep": 0.25, "residual": 2.0, "parallax": 0.0},
    "v6_kkt_reliable25_res3_parallax05": {"keep": 0.25, "residual": 3.0, "parallax": 0.5},
    "v7_kkt_reliable25_res3_parallax1": {"keep": 0.25, "residual": 3.0, "parallax": 1.0},
}


def generate(base_path: Path, suite_dir: Path) -> dict[str, Any]:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Expected YAML mapping in {base_path}.")
    config_dir = suite_dir / "configs"
    result_dir = suite_dir / "results"
    config_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"base_config": str(base_path.resolve()), "variants": []}
    for name, variant in VARIANTS.items():
        config = copy.deepcopy(base)
        config["experiment_name"] = name
        config["ba"].update(
            {
                "outer_schedule": [True, False, False],
                "dense_depth_mode": "none",
                "gauge_mode": "initial_baseline",
                "solver_mode": "standard_lm",
                "max_initial_residual_deg": variant["residual"],
                "min_parallax_deg": variant["parallax"],
            }
        )
        config["matching"]["reliability_keep_fraction"] = variant["keep"]
        config_path = config_dir / f"{name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        manifest["variants"].append(
            {
                "name": name,
                "config": str(config_path),
                "result": str(result_dir / f"{name}.json"),
                **variant,
            }
        )
    (suite_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.base, args.suite_dir), indent=2))


if __name__ == "__main__":
    main()
