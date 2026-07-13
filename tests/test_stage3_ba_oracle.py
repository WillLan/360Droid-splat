from __future__ import annotations

import torch

from geometry.spherical_erp import erp_pixel_to_unit_ray
from models.spherical_selfi_stage3_ba import Stage3MatchCache, all_directed_pairs
from tools.evaluate_stage3_ba_oracle import build_gt_oracle_match_cache, evaluate_oracle
from training.train_spherical_ba_recurrent_refiner import default_config


def test_gt_oracle_cache_replaces_only_target_geometry() -> None:
    batch, views, queries, height, width = 1, 2, 6, 8, 16
    base_uv = torch.tensor(
        [
            [1.5, 1.5],
            [4.5, 2.5],
            [7.5, 3.5],
            [10.5, 4.5],
            [13.5, 5.5],
            [15.5, 6.5],
        ]
    )
    source_uv = base_uv.view(1, 1, queries, 2).repeat(batch, views, 1, 1)
    source_ray = erp_pixel_to_unit_ray(source_uv, height, width)
    edges = all_directed_pairs(views)
    edge_count = int(edges.shape[0])
    shape = (batch, edge_count, queries)
    factor_weight = torch.linspace(0.2, 0.9, edge_count * queries).reshape(shape)
    reference = Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=torch.full((batch, views, queries), 2.0),
        source_valid=torch.ones(batch, views, queries, dtype=torch.bool),
        edges=edges,
        target_uv=torch.zeros(batch, edge_count, queries, 2),
        target_ray=-source_ray[:, edges[:, 0]],
        top1_cosine=torch.ones(shape),
        top2_margin=torch.ones(shape),
        entropy=torch.zeros(shape),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        factor_weight=factor_weight,
    )
    gt_depth = torch.full((batch, views, 1, height, width), 2.0)
    gt_valid = torch.ones_like(gt_depth, dtype=torch.bool)
    gt_pose = torch.eye(4).view(1, 1, 4, 4).repeat(batch, views, 1, 1)
    gt_pose[:, 1, 0, 3] = 0.1

    oracle = build_gt_oracle_match_cache(
        reference,
        gt_depth,
        gt_valid,
        gt_pose,
        depth_consistency_abs=10.0,
        depth_consistency_rel=0.0,
    )

    source_index, target_index = (int(value) for value in edges[0])
    point_source = 2.0 * source_ray[0, source_index]
    point_world = point_source + gt_pose[0, source_index, :3, 3]
    point_target = point_world - gt_pose[0, target_index, :3, 3]
    expected = torch.nn.functional.normalize(point_target, dim=-1)
    torch.testing.assert_close(oracle.target_ray[0, 0], expected, atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(oracle.source_uv, reference.source_uv)
    torch.testing.assert_close(oracle.factor_weight, reference.factor_weight)
    assert bool(oracle.valid_mask.all())
    assert oracle.metadata["oracle"] is True


def test_synthetic_gt_oracle_evaluator_runs_all_arms() -> None:
    config = default_config()
    config["stage3"]["enabled"] = True
    config["matching"].update(
        {
            "num_queries": 2,
            "query_chunk_size": 1,
            "forward_backward": False,
            "min_factor_weight": 0.0,
            "reliability_keep_fraction": 1.0,
        }
    )
    config["ba"].update(
        {
            "iterations": 0,
            "min_factors": 1,
            "min_affine_support": 2,
            "factor_chunk_size": 4,
        }
    )
    config["dataset"]["max_val_samples"] = 1
    config["train"].update(
        {
            "batch_size": 1,
            "num_workers": 0,
            "feature_device": "cpu",
            "train_device": "cpu",
            "amp": False,
        }
    )
    result = evaluate_oracle(
        config,
        max_batches=1,
        diagnostic_iterations=1,
        diagnostic_max_pose_update_deg=0.5,
    )
    assert result["format"] == "spherical_stage3_ba_gt_oracle_evaluation_v1"
    assert result["num_batches"] == 1
    for solver in ("formal", "diagnostic"):
        for arm in ("adapter_stage2", "gt_stage2", "gt_gtdepth"):
            assert f"{solver}/{arm}/pose_rotation_mean_deg" in result["mean"]
