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

CYCLE_CONSISTENCY_VARIANTS: dict[str, dict[str, Any]] = {
    "c0_rotation_fb1": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 1.0},
    "c1_rotation_fb05": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 0.5},
    "c2_rotation_fb025": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 0.25},
    "c3_rotation_fb01": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 0.1},
    "c4_rotation_fb05_reliable50": {"keep": 0.5, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 0.5},
    "c5_rotation_fb025_reliable50": {"keep": 0.5, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "fb": 0.25},
    "c6_se3_fb05_trans001": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "fb": 0.5, "translation": 0.001},
    "c7_se3_fb025_trans001": {"keep": 1.0, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "fb": 0.25, "translation": 0.001},
}

RESIDUAL_TRIGGER_VARIANTS: dict[str, dict[str, Any]] = {
    "g0_se3_reliable10_gate0": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "trigger": 0.0},
    "g1_se3_reliable10_gate026": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "trigger": 0.26},
    "g2_se3_reliable10_gate028": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "trigger": 0.28},
    "g3_se3_reliable10_gate030": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "trigger": 0.30},
    "g4_rotation_reliable10_gate026": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.26},
    "g5_rotation_reliable10_gate028": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.28},
    "g6_rotation_reliable10_gate030": {"keep": 0.1, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.30},
    "g7_rotation_reliable05_gate026": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.26},
    "g8_rotation_reliable05_gate028": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.28},
    "g9_rotation_reliable05_gate030": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "trigger": 0.30},
    "g10_se3_reliable05_gate028": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "trigger": 0.28},
}

DISTINCTIVENESS_VARIANTS: dict[str, dict[str, Any]] = {
    "d0_rotation_reliable05_exclusion0": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 0.0},
    "d1_rotation_reliable05_exclusion1": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 1.0},
    "d2_rotation_reliable05_exclusion2": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 2.0},
    "d3_rotation_reliable05_exclusion5": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 5.0},
    "d4_rotation_reliable05_exclusion10": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 10.0},
    "d5_rotation_reliable10_exclusion2": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 2.0},
    "d6_rotation_reliable10_exclusion5": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "exclusion": 5.0},
    "d7_se3_reliable05_exclusion2": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "exclusion": 2.0},
    "d8_se3_reliable05_exclusion5": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "exclusion": 5.0},
}

ROTATION_TRUST_VARIANTS: dict[str, dict[str, Any]] = {
    "q0_rotation_reliable05_rot5": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "rotation": 5.0},
    "q1_rotation_reliable05_rot02": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "rotation": 0.2},
    "q2_rotation_reliable05_rot01": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "rotation": 0.1},
    "q3_rotation_reliable05_rot005": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "rotation": 0.05},
    "q4_rotation_reliable05_rot002": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "rotation": 0.02},
    "q5_se3_reliable05_rot02": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "rotation": 0.2},
    "q6_se3_reliable05_rot01": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "rotation": 0.1},
    "q7_se3_reliable05_rot005": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "rotation": 0.05},
    "q8_se3_reliable10_rot01": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "rotation": 0.1},
}

SUBPIXEL_VARIANTS: dict[str, dict[str, Any]] = {
    "s0_rotation_reliable05_radius0": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "subpixel": 0},
    "s1_rotation_reliable05_radius1": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "subpixel": 1},
    "s2_rotation_reliable05_radius2": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "subpixel": 2},
    "s3_rotation_reliable10_radius1": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "subpixel": 1},
    "s4_rotation_reliable10_radius2": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "rotation_only", "subpixel": 2},
    "s5_se3_reliable05_radius1": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "subpixel": 1},
    "s6_se3_reliable05_radius2": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "se3", "translation": 0.001, "subpixel": 2},
}

TRANSLATION_ONLY_VARIANTS: dict[str, dict[str, Any]] = {
    "u0_translation_reliable10_trans001": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "translation_only", "translation": 0.001},
    "u1_translation_reliable05_trans001": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "translation_only", "translation": 0.001},
    "u2_translation_reliable10_trans0005": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "translation_only", "translation": 0.0005},
    "u3_translation_reliable10_trans002": {"keep": 0.10, "residual": None, "parallax": 0.0, "side": "right", "dof": "translation_only", "translation": 0.002},
    "u4_translation_reliable05_trans0005": {"keep": 0.05, "residual": None, "parallax": 0.0, "side": "right", "dof": "translation_only", "translation": 0.0005},
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
        if "rotation" in variant:
            config["ba"]["max_pose_update_deg"] = variant["rotation"]
        if "iterations" in variant:
            config["ba"]["iterations"] = variant["iterations"]
        if "fb" in variant:
            config["matching"]["forward_backward"] = True
            config["matching"]["fb_tolerance_deg"] = variant["fb"]
        if "trigger" in variant:
            config["ba"]["min_initial_median_residual_deg"] = variant["trigger"]
        if "exclusion" in variant:
            config["matching"]["distinctiveness_exclusion_deg"] = variant["exclusion"]
        if "subpixel" in variant:
            config["matching"]["subpixel_refine_radius"] = variant["subpixel"]
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
        choices=("reliability", "high_parallax", "trust_region", "rotation_only", "cycle_consistency", "residual_trigger", "distinctiveness", "rotation_trust", "subpixel", "translation_only"),
        default="reliability",
    )
    args = parser.parse_args()
    variants = {
        "reliability": VARIANTS,
        "high_parallax": HIGH_PARALLAX_VARIANTS,
        "trust_region": TRUST_REGION_VARIANTS,
        "rotation_only": ROTATION_ONLY_VARIANTS,
        "cycle_consistency": CYCLE_CONSISTENCY_VARIANTS,
        "residual_trigger": RESIDUAL_TRIGGER_VARIANTS,
        "distinctiveness": DISTINCTIVENESS_VARIANTS,
        "rotation_trust": ROTATION_TRUST_VARIANTS,
        "subpixel": SUBPIXEL_VARIANTS,
        "translation_only": TRANSLATION_ONLY_VARIANTS,
    }[args.profile]
    print(json.dumps(generate(args.base, args.suite_dir, variants=variants), indent=2))


if __name__ == "__main__":
    main()
