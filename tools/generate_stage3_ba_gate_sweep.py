"""Generate fixed-window BA-only factor-gating sweep configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


VARIANTS: dict[str, dict[str, Any]] = {
    "v0_kkt_all": {"keep": 1.0, "residual": None, "parallax": 0.0},
    "v1_kkt_reliable50": {"keep": 0.50, "residual": None, "parallax": 0.0},
    "v2_kkt_reliable25": {"keep": 0.25, "residual": None, "parallax": 0.0},
    "v3_kkt_reliable10": {"keep": 0.10, "residual": None, "parallax": 0.0},
    "v4_kkt_reliable25_res3": {"keep": 0.25, "residual": 3.0, "parallax": 0.0},
    "v5_kkt_reliable25_res2": {"keep": 0.25, "residual": 2.0, "parallax": 0.0},
    "v6_kkt_reliable25_res3_parallax05": {"keep": 0.25, "residual": 3.0, "parallax": 0.5},
    "v7_kkt_reliable25_res3_parallax1": {"keep": 0.25, "residual": 3.0, "parallax": 1.0},
}

HIGH_PARALLAX_VARIANTS: dict[str, dict[str, Any]] = {
    "p0_kkt_all_parallax2": {"keep": 1.0, "residual": None, "parallax": 2.0},
    "p1_kkt_all_parallax3": {"keep": 1.0, "residual": None, "parallax": 3.0},
    "p2_kkt_all_parallax5": {"keep": 1.0, "residual": None, "parallax": 5.0},
    "p3_kkt_reliable50_parallax2": {"keep": 0.50, "residual": None, "parallax": 2.0},
    "p4_kkt_reliable50_parallax3": {"keep": 0.50, "residual": None, "parallax": 3.0},
    "p5_kkt_reliable50_parallax5": {"keep": 0.50, "residual": None, "parallax": 5.0},
    "p6_kkt_reliable25_parallax2": {"keep": 0.25, "residual": None, "parallax": 2.0},
    "p7_kkt_reliable25_parallax3": {"keep": 0.25, "residual": None, "parallax": 3.0},
    "p8_kkt_reliable25_parallax5": {"keep": 0.25, "residual": None, "parallax": 5.0},
}

TRUST_REGION_VARIANTS: dict[str, dict[str, Any]] = {
    "t0_left_all_trans01": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "left", "translation": 0.01},
    "t1_left_all_trans005": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "left", "translation": 0.005},
    "t2_left_reliable10_trans01": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "left", "translation": 0.01},
    "t3_left_reliable10_trans005": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "left", "translation": 0.005},
    "t4_right_all_trans05": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.05},
    "t5_right_all_trans01": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.01},
    "t6_right_all_trans005": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.005},
    "t7_right_reliable50_trans01": {"keep": 0.5, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.01},
    "t8_right_reliable25_trans01": {"keep": 0.25, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.01},
    "t9_right_reliable10_trans01": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.01},
    "t10_right_reliable10_trans005": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.005},
    "t11_right_reliable10_trans001": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.001},
    "t12_right_reliable10_iter1": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "translation": 0.05, "iterations": 1},
}

ROTATION_ONLY_VARIANTS: dict[str, dict[str, Any]] = {
    "r0_rotation_all": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
    "r1_rotation_reliable75": {"keep": 0.75, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
    "r2_rotation_reliable50": {"keep": 0.50, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
    "r3_rotation_reliable25": {"keep": 0.25, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
    "r4_rotation_reliable10": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
    "r5_rotation_reliable05": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only"},
}


def generate(
    base_path: Path,
    suite_dir: Path,
    *,
    variants: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Expected YAML mapping in {base_path}.")
    config_dir = suite_dir / "configs"
    result_dir = suite_dir / "results"
    config_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {"base_config": str(base_path.resolve()), "variants": []}
    selected_variants = VARIANTS if variants is None else variants
    for name, variant in selected_variants.items():
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
        if "side" in variant:
            config["ba"]["pose_update_side"] = variant["side"]
        if "dof" in variant:
            config["ba"]["pose_dof_mode"] = variant["dof"]
        if "translation" in variant:
            config["ba"]["max_translation_update"] = variant["translation"]
        if "iterations" in variant:
            config["ba"]["iterations"] = variant["iterations"]
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
    parser.add_argument(
        "--profile",
        choices=("reliability", "high_parallax", "trust_region", "rotation_only"),
        default="reliability",
    )
    args = parser.parse_args()
    variants = {
        "reliability": VARIANTS,
        "high_parallax": HIGH_PARALLAX_VARIANTS,
        "trust_region": TRUST_REGION_VARIANTS,
        "rotation_only": ROTATION_ONLY_VARIANTS,
    }[args.profile]
    print(json.dumps(generate(args.base, args.suite_dir, variants=variants), indent=2))


if __name__ == "__main__":
    main()
