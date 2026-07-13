from __future__ import annotations

import torch

from geometry.spherical_erp import erp_pixel_to_unit_ray
from models.spherical_selfi_stage3_ba import Stage3MatchCache, all_directed_pairs
from tools.evaluate_stage3_ba_star import (
    STAR_VARIANTS,
    evaluate_star,
    select_match_cache_variant,
    topology_edge_mask,
)
from training.train_spherical_ba_recurrent_refiner import default_config


def _cache(*, queries: int = 10) -> Stage3MatchCache:
    batch, views, height, width = 1, 4, 8, 16
    source_uv = torch.zeros(batch, views, queries, 2)
    source_uv[..., 0] = torch.arange(queries).remainder(width).float() + 0.5
    source_uv[..., 1] = 3.5
    source_ray = erp_pixel_to_unit_ray(source_uv, height, width)
    edges = all_directed_pairs(views)
    shape = (batch, int(edges.shape[0]), queries)
    factor_weight = torch.arange(1, 1 + queries).float().view(1, 1, queries).expand(shape)
    return Stage3MatchCache(
        source_uv=source_uv,
        source_ray=source_ray,
        source_depth=torch.full((batch, views, queries), 2.0),
        source_valid=torch.ones(batch, views, queries, dtype=torch.bool),
        edges=edges,
        target_uv=torch.zeros(*shape, 2),
        target_ray=torch.nn.functional.normalize(torch.ones(*shape, 3), dim=-1),
        top1_cosine=torch.ones(shape),
        top2_margin=torch.ones(shape),
        entropy=torch.zeros(shape),
        valid_mask=torch.ones(shape, dtype=torch.bool),
        factor_weight=factor_weight,
        metadata={"raw": True},
    )


def test_star_topology_masks_select_expected_edges() -> None:
    edges = all_directed_pairs(4)
    forward = edges[topology_edge_mask(edges, "star_forward")].tolist()
    bidirectional = edges[topology_edge_mask(edges, "star_bidirectional")].tolist()
    assert forward == [[0, 1], [0, 2], [0, 3]]
    assert bidirectional == [[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]]
    assert int(topology_edge_mask(edges, "all_directed").sum()) == 12


def test_star_variants_share_raw_cache_and_equalize_factor_budgets() -> None:
    raw = _cache(queries=10)
    expected_counts = {
        "a_all_top10": 12,
        "b_star_top10": 3,
        "c_star_top40": 12,
        "d_star_bidir_top20": 12,
    }
    for name, variant_cfg in STAR_VARIANTS.items():
        selected = select_match_cache_variant(raw, **variant_cfg)
        assert int(selected.valid_mask.sum()) == expected_counts[name]
        assert selected.source_uv is raw.source_uv
        assert selected.source_ray is raw.source_ray
        assert selected.target_ray is raw.target_ray
        assert selected.factor_weight is raw.factor_weight
        assert selected.valid_mask is not raw.valid_mask
        assert selected.metadata["raw"] is True
        assert selected.metadata["star_evaluation_shared_raw_cache"] is True


def test_star_selection_keeps_highest_weight_per_edge() -> None:
    raw = _cache(queries=10)
    selected = select_match_cache_variant(
        raw,
        topology="star_forward",
        keep_fraction=0.4,
    )
    for edge_index, pair in enumerate(raw.edges.tolist()):
        indices = torch.nonzero(selected.valid_mask[0, edge_index], as_tuple=False).flatten()
        if pair[0] == 0:
            assert indices.tolist() == [6, 7, 8, 9]
        else:
            assert int(indices.numel()) == 0


def test_synthetic_star_evaluator_runs_all_variants() -> None:
    config = default_config()
    config["stage3"]["enabled"] = True
    config["matching"].update(
        {
            "num_queries": 10,
            "query_chunk_size": 2,
            "forward_backward": False,
            "min_factor_weight": 0.0,
            "reliability_keep_fraction": 0.1,
        }
    )
    config["ba"].update(
        {
            "iterations": 0,
            "min_factors": 1,
            "min_affine_support": 2,
            "factor_chunk_size": 16,
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
    result = evaluate_star(config, max_batches=1, start_batch=0)
    assert result["format"] == "spherical_stage3_ba_star_evaluation_v1"
    assert result["num_batches"] == 1
    assert result["shared_raw_matching"] is True
    for name in STAR_VARIANTS:
        assert f"{name}/valid_factors" in result["mean"]
        assert f"{name}/pose_rotation_mean_deg" in result["mean"]
        assert f"{name}/profile_ba_sec" in result["mean"]
