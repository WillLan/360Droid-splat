import torch
import pytest

from backend.pano_gs.mapper import PanoGaussianMap, PanoGaussianMapper
from frontend.pano_droid.interfaces import FrontendOutput
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from mapping.gaussian_initializer import GaussianInitializer, GaussianSeedBatch


def test_gaussian_initializer_backprojects_keyframe():
    H, W = 8, 16
    output = FrontendOutput(
        frame_id=2,
        timestamp=2.0,
        pose_c2w=torch.eye(4),
        relative_pose=torch.eye(4),
        pose_confidence=0.9,
        inverse_depth=torch.full((1, H, W), 0.5),
        depth_confidence=torch.ones(1, H, W),
        spherical_flow=torch.zeros(2, H, W),
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked",
    )
    image = torch.rand(3, H, W)
    initializer = GaussianInitializer(max_seeds_per_keyframe=10)
    seeds = initializer.from_frontend_output(output, image)
    assert len(seeds) == 10
    assert seeds.xyz.shape == (10, 3)
    assert torch.isfinite(seeds.xyz).all()
    assert torch.all((seeds.rgb >= 0.0) & (seeds.rgb <= 1.0))


def test_gaussian_initializer_dense_backprojects_every_valid_pixel():
    H, W = 4, 8
    inv = torch.full((1, H, W), 0.25)
    conf = torch.ones(1, H, W)
    inv[0, 0, 0] = 0.0
    conf[0, 1, 1] = 0.1
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    output = FrontendOutput(
        frame_id=3,
        timestamp=3.0,
        pose_c2w=pose,
        relative_pose=torch.eye(4),
        pose_confidence=0.9,
        inverse_depth=inv,
        depth_confidence=conf,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked",
    )
    image = torch.rand(3, H, W)
    initializer = GaussianInitializer(max_seeds_per_keyframe=0, min_confidence=0.5)
    seeds = initializer.from_frontend_output(output, image)

    valid = (inv > 1e-6) & (conf >= 0.5)
    assert len(seeds) == int(valid.sum())
    assert seeds.source_hw == (H, W)
    assert torch.equal(seeds.source_flat_idx.cpu(), torch.nonzero(valid.view(-1), as_tuple=False).flatten())

    row, col = 2, 3
    flat = row * W + col
    kept = torch.nonzero(valid.view(-1), as_tuple=False).flatten()
    seed_idx = int(torch.nonzero(kept == flat, as_tuple=False).flatten()[0])
    pixel = pixel_grid(H, W)[row, col]
    bearing = erp_pixel_to_bearing(pixel, H, W)
    expected = pose[:3, 3] + bearing * 4.0
    assert torch.allclose(seeds.xyz[seed_idx], expected, atol=1e-5)


def test_gaussian_initializer_sky_mask_skips_blue_upper_pixels():
    H, W = 6, 8
    output = FrontendOutput(
        frame_id=4,
        timestamp=4.0,
        pose_c2w=torch.eye(4),
        relative_pose=torch.eye(4),
        pose_confidence=0.9,
        inverse_depth=torch.full((1, H, W), 0.5),
        depth_confidence=torch.ones(1, H, W),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked",
    )
    image = torch.zeros(3, H, W)
    image[0] = 0.25
    image[1] = 0.25
    image[2] = 0.25
    image[:, :3, :] = torch.tensor([0.10, 0.35, 0.90]).view(3, 1, 1)
    initializer = GaussianInitializer(
        max_seeds_per_keyframe=0,
        sky_mask_enable=True,
        sky_mask_top_ratio=0.5,
    )
    seeds = initializer.from_frontend_output(output, image)

    assert len(seeds) == (H - 3) * W
    assert torch.allclose(seeds.rgb, torch.full_like(seeds.rgb, 0.25))


def test_gaussian_initializer_panovggt_sky_mask_source_requires_hint():
    H, W = 3, 4
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    points = torch.stack([xx.float(), yy.float(), torch.ones(H, W)], dim=-1)
    conf = torch.ones(1, H, W)
    output = FrontendOutput(
        frame_id=40,
        timestamp=40.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked",
        world_points=points,
        world_points_confidence=conf,
        valid_world_points_mask=torch.ones(1, H, W, dtype=torch.bool),
    )
    image = torch.rand(3, H, W)
    initializer = GaussianInitializer(
        max_seeds_per_keyframe=0,
        seed_source="world_points_only",
        sky_mask_enable=True,
        sky_mask_source="panovggt_head",
        insertion_strategy="pfgs360_replace_fuse",
    )

    with pytest.raises(ValueError, match="requires insertion_hints"):
        initializer.from_frontend_output(output, image)

    sky_mask = torch.zeros(1, H, W, dtype=torch.bool)
    sky_mask[0, 0, 0] = True
    seeds = initializer.from_frontend_output(
        output,
        image,
        insertion_hints={"sky_mask": sky_mask},
        first_keyframe=True,
    )

    assert len(seeds) == H * W - 1
    assert 0 not in set(int(v) for v in seeds.source_flat_idx.tolist())


def test_gaussian_initializer_world_points_only_uses_input_xyz_exactly():
    H, W = 3, 4
    points = torch.arange(H * W * 3, dtype=torch.float32).view(H, W, 3)
    conf = torch.ones(1, H, W)
    valid = torch.ones(1, H, W, dtype=torch.bool)
    valid[0, 0, 0] = False
    output = FrontendOutput(
        frame_id=5,
        timestamp=5.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
        world_points=points,
        world_points_confidence=conf,
        valid_world_points_mask=valid,
    )
    image = torch.rand(3, H, W)
    initializer = GaussianInitializer(max_seeds_per_keyframe=0, seed_source="world_points_only")
    seeds = initializer.from_frontend_output(output, image)
    expected = points.reshape(-1, 3)[valid.view(-1)]
    assert torch.allclose(seeds.xyz, expected)
    assert seeds.source_hw == (H, W)
    assert torch.equal(seeds.source_flat_idx.cpu(), torch.nonzero(valid.view(-1), as_tuple=False).flatten())


def test_gaussian_initializer_world_points_only_requires_world_points():
    output = FrontendOutput(
        frame_id=6,
        timestamp=6.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, 2, 4),
        depth_confidence=torch.ones(1, 2, 4),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
    )
    initializer = GaussianInitializer(seed_source="world_points_only")
    try:
        initializer.from_frontend_output(output, torch.rand(3, 2, 4))
    except ValueError as exc:
        assert "world_points" in str(exc)
    else:
        raise AssertionError("world_points_only should reject missing world_points")


def test_pfgs360_initializer_uses_single_voxel_and_aggregates_candidates():
    H, W = 1, 4
    points = torch.tensor(
        [
            [[0.01, 0.0, 1.0], [0.05, 0.0, 1.0], [0.25, 0.0, 1.0], [0.37, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    conf = torch.tensor([[[0.2, 0.8, 1.0, 0.5]]], dtype=torch.float32)
    valid = torch.ones(1, H, W, dtype=torch.bool)
    output = FrontendOutput(
        frame_id=8,
        timestamp=8.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
        world_points=points,
        world_points_confidence=conf,
        valid_world_points_mask=valid,
    )
    hints = {
        "non_sky": torch.ones(1, H, W, dtype=torch.bool),
        "pair_confidence": torch.tensor([[[0.9, 0.6, 0.8, 0.1]]], dtype=torch.float32),
    }
    initializer = GaussianInitializer(
        max_seeds_per_keyframe=0,
        seed_source="world_points_only",
        insertion_strategy="pfgs360",
        pfgs360_voxel_size=0.12,
        temporal_pair_conf_min=0.7,
    )

    seeds = initializer.from_frontend_output(output, torch.rand(3, H, W), insertion_hints=hints, first_keyframe=False)

    assert len(seeds) == 3
    assert torch.allclose(seeds.scale, torch.full_like(seeds.scale, 0.12))
    assert torch.equal(seeds.level.cpu(), torch.zeros(3, dtype=torch.int8))
    assert seeds.grid_coord is not None
    assert torch.equal(seeds.grid_coord.cpu(), torch.tensor([[0, 0, 8], [2, 0, 8], [3, 0, 8]], dtype=torch.int32))
    assert seeds.insert_enabled is not None
    assert torch.equal(seeds.insert_enabled.cpu(), torch.tensor([True, True, False]))
    assert seeds.source_flat_idx is not None
    assert torch.equal(seeds.source_flat_idx.cpu(), torch.tensor([1, 2, 3]))


def test_replace_fuse_initializer_keeps_dense_candidates_without_pair_conf_gate():
    H, W = 1, 4
    points = torch.tensor(
        [
            [[0.01, 0.0, 1.0], [0.05, 0.0, 1.0], [0.25, 0.0, 1.0], [0.37, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    output = FrontendOutput(
        frame_id=8,
        timestamp=8.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
        world_points=points,
        world_points_confidence=torch.ones(1, H, W),
        valid_world_points_mask=torch.ones(1, H, W, dtype=torch.bool),
    )
    hints = {
        "non_sky": torch.ones(1, H, W, dtype=torch.bool),
        "pair_confidence": torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32),
    }
    initializer = GaussianInitializer(
        max_seeds_per_keyframe=0,
        seed_source="world_points_only",
        insertion_strategy="pfgs360_replace_fuse",
        pfgs360_voxel_size=0.12,
        temporal_pair_conf_min=0.7,
    )

    seeds = initializer.from_frontend_output(output, torch.rand(3, H, W), insertion_hints=hints, first_keyframe=False)

    assert len(seeds) == W
    assert seeds.grid_coord is not None
    assert torch.equal(seeds.grid_coord.cpu(), torch.floor(points.reshape(-1, 3) / 0.12).to(torch.int32))
    assert torch.equal(seeds.insert_enabled.cpu(), torch.ones(W, dtype=torch.bool))
    assert torch.equal(seeds.source_flat_idx.cpu(), torch.arange(W, dtype=torch.long))


def test_replace_fuse_initializer_decouples_voxel_and_erp_depth_latitude_scale():
    H, W = 8, 16
    grid = pixel_grid(H, W, device=torch.device("cpu"), dtype=torch.float32).view(-1, 2)
    points = erp_pixel_to_bearing(grid, H, W).view(H, W, 3) * 0.05
    conf = torch.ones(1, H, W, dtype=torch.float32)
    valid = torch.ones(1, H, W, dtype=torch.bool)
    output = FrontendOutput(
        frame_id=9,
        timestamp=9.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
        world_points=points,
        world_points_confidence=conf,
        valid_world_points_mask=valid,
    )
    initializer = GaussianInitializer(
        max_seeds_per_keyframe=0,
        seed_source="world_points_only",
        insertion_strategy="pfgs360_replace_fuse",
        pfgs360_voxel_size=0.02,
        pfgs360_gaussian_scale_mode="erp_depth_latitude",
        pfgs360_gaussian_scale_factor=1.25,
        pfgs360_gaussian_scale_min=0.008,
        pfgs360_gaussian_scale_max=0.08,
        pfgs360_gaussian_scale_lat_cos_min=0.25,
    )

    seeds = initializer.from_frontend_output(output, torch.rand(3, H, W), first_keyframe=True)

    assert seeds.grid_coord is not None
    assert torch.equal(seeds.grid_coord.cpu(), torch.floor(seeds.xyz.detach().cpu() / 0.02).to(torch.int32))
    assert torch.all(seeds.scale >= 0.008)
    assert torch.all(seeds.scale <= 0.08)
    assert not torch.allclose(seeds.scale, torch.full_like(seeds.scale, 0.02))
    top_scale = seeds.scale[seeds.source_flat_idx.cpu() // W == 0].mean()
    mid_scale = seeds.scale[seeds.source_flat_idx.cpu() // W == H // 2].mean()
    assert float(mid_scale) > float(top_scale)


def _seed_batch(xyz: torch.Tensor, confidence: torch.Tensor | None = None, *, frame_id: int = 0) -> GaussianSeedBatch:
    n = int(xyz.shape[0])
    if confidence is None:
        confidence = torch.ones(n)
    return GaussianSeedBatch(
        xyz=xyz.float(),
        rgb=torch.full((n, 3), 0.5),
        confidence=confidence.float(),
        scale=torch.ones(n),
        level=torch.zeros(n, dtype=torch.int8),
        frame_id=frame_id,
    )


def _pfgs_seed_batch(
    xyz: torch.Tensor,
    *,
    frame_id: int,
    insert_enabled: torch.Tensor | None = None,
    insert_score: torch.Tensor | None = None,
    source_flat_idx: torch.Tensor | None = None,
    source_hw: tuple[int, int] | None = None,
) -> GaussianSeedBatch:
    n = int(xyz.shape[0])
    return GaussianSeedBatch(
        xyz=xyz.float(),
        rgb=torch.full((n, 3), 0.5),
        confidence=torch.ones(n),
        scale=torch.full((n,), 0.1),
        level=torch.zeros(n, dtype=torch.int8),
        frame_id=frame_id,
        source_flat_idx=source_flat_idx,
        source_hw=source_hw,
        insert_enabled=insert_enabled,
        insert_score=insert_score,
    )


def _frontend_output_for_mapper(frame_id: int) -> FrontendOutput:
    return FrontendOutput(
        frame_id=frame_id,
        timestamp=float(frame_id),
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=None,
        depth_confidence=None,
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=0.0,
        tracking_status="tracked_panovggt_long",
    )


def test_novel_gaussian_insertion_obeys_keyframe_and_global_budgets():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "first_keyframe_max_seeds": 2,
                "keyframe_max_seeds": 2,
                "global_anchor_budget": 3,
                "voxel_neighbor_radius": 0,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    first = _seed_batch(
        torch.tensor([[0.1, 0.0, 0.0], [2.1, 0.0, 0.0], [4.1, 0.0, 0.0], [6.1, 0.0, 0.0]]),
        torch.tensor([0.1, 0.9, 0.8, 0.7]),
    )
    assert mapper.insert_keyframe(first, _frontend_output_for_mapper(0)) == 2
    assert mapper.map.anchor_count() == 2
    assert mapper.stats.last_skipped_budget == 2

    second = _seed_batch(torch.tensor([[8.1, 0.0, 0.0], [10.1, 0.0, 0.0]]), frame_id=1)
    assert mapper.insert_keyframe(second, _frontend_output_for_mapper(1)) == 1
    assert mapper.map.anchor_count() == 3
    assert mapper.stats.last_skipped_budget == 1


def test_mapper_tracks_requested_and_inserted_seed_source_pixels():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "first_keyframe_max_seeds": 2,
                "keyframe_max_seeds": 2,
                "global_anchor_budget": 10,
                "voxel_neighbor_radius": 0,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    seeds = _seed_batch(
        torch.tensor([[0.1, 0.0, 0.0], [2.1, 0.0, 0.0], [4.1, 0.0, 0.0]]),
        torch.tensor([0.2, 0.9, 0.8]),
    )
    seeds.source_flat_idx = torch.tensor([3, 5, 7], dtype=torch.long)
    seeds.source_hw = (2, 4)

    assert mapper.insert_keyframe(seeds, _frontend_output_for_mapper(0)) == 2

    assert mapper.last_source_hw == (2, 4)
    assert torch.equal(mapper.last_requested_source_flat_idx, torch.tensor([3, 5, 7]))
    assert torch.equal(mapper.last_inserted_source_flat_idx, torch.tensor([5, 7]))


def test_novel_gaussian_insertion_voxel_dedup_updates_existing_observation():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "voxel_neighbor_radius": 0,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    assert mapper.insert_keyframe(_seed_batch(torch.tensor([[0.1, 0.0, 0.0]])), _frontend_output_for_mapper(0)) == 1
    before_obs = int(mapper.map._anchor_obs_count[0])
    before_conf = float(mapper.map._anchor_conf_accum[0])

    duplicate = _seed_batch(torch.tensor([[0.2, 0.0, 0.0], [2.1, 0.0, 0.0]]), torch.tensor([0.6, 0.9]), frame_id=1)
    assert mapper.insert_keyframe(duplicate, _frontend_output_for_mapper(1)) == 1

    assert mapper.map.anchor_count() == 2
    assert mapper.stats.last_skipped_voxel == 1
    assert int(mapper.map._anchor_obs_count[0]) == before_obs + 1
    assert float(mapper.map._anchor_conf_accum[0]) > before_conf


def test_novel_gaussian_insertion_updates_existing_voxel_when_global_budget_full():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 1,
                "voxel_neighbor_radius": 0,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    assert mapper.insert_keyframe(_seed_batch(torch.tensor([[0.1, 0.0, 0.0]])), _frontend_output_for_mapper(0)) == 1
    before_obs = int(mapper.map._anchor_obs_count[0])

    seeds = _seed_batch(torch.tensor([[0.2, 0.0, 0.0], [2.1, 0.0, 0.0]]), torch.tensor([0.6, 0.9]), frame_id=1)
    assert mapper.insert_keyframe(seeds, _frontend_output_for_mapper(1)) == 0

    assert mapper.map.anchor_count() == 1
    assert mapper.stats.last_skipped_voxel == 1
    assert mapper.stats.last_skipped_budget == 1
    assert int(mapper.map._anchor_obs_count[0]) == before_obs + 1


def test_novel_gaussian_insertion_uses_separate_first_keyframe_neighbor_radius():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "first_keyframe_voxel_neighbor_radius": 0,
                "voxel_neighbor_radius": 1,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    first = _seed_batch(torch.tensor([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]), frame_id=0)
    assert mapper.insert_keyframe(first, _frontend_output_for_mapper(0)) == 2
    assert mapper.map.anchor_count() == 2

    near_existing = _seed_batch(torch.tensor([[2.1, 0.0, 0.0]]), frame_id=1)
    assert mapper.insert_keyframe(near_existing, _frontend_output_for_mapper(1)) == 0
    assert mapper.map.anchor_count() == 2
    assert mapper.stats.last_skipped_voxel == 1


def test_pfgs360_mapper_hash_hits_near_hits_and_suppressed_misses():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360",
                "voxel_size": 0.1,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "near_grid_radius": 1,
                "near_distance_factor": 1.0,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    first = _pfgs_seed_batch(torch.tensor([[0.05, 0.0, 0.0]]), frame_id=0)
    assert mapper.insert_keyframe(first, _frontend_output_for_mapper(0)) == 1
    before_obs = int(mapper.map._anchor_obs_count[0])

    second = _pfgs_seed_batch(
        torch.tensor([[0.06, 0.0, 0.0], [0.14, 0.0, 0.0], [0.35, 0.0, 0.0], [0.55, 0.0, 0.0]]),
        frame_id=1,
        insert_enabled=torch.tensor([True, True, False, True]),
        insert_score=torch.tensor([0.9, 0.8, 1.0, 0.7]),
    )

    assert mapper.insert_keyframe(second, _frontend_output_for_mapper(1)) == 1

    assert mapper.map.anchor_count() == 2
    assert mapper.stats.last_hash_hits == 1
    assert mapper.stats.last_hash_near_hits == 1
    assert mapper.stats.last_skipped_voxel == 2
    assert mapper.stats.last_suppressed_insert == 1
    assert int(mapper.map._anchor_obs_count[0]) == before_obs + 2


def test_replace_fuse_refreshes_voxel_from_current_xyz_before_occupancy_reject():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    first = _pfgs_seed_batch(torch.tensor([[0.005, 0.0, 0.0]]), frame_id=0)
    assert mapper.insert_keyframe(first, _frontend_output_for_mapper(0)) == 1
    with torch.no_grad():
        mapper.map.xyz.data[0] = torch.tensor([0.031, 0.0, 0.0])

    second = _pfgs_seed_batch(torch.tensor([[0.032, 0.0, 0.0]]), frame_id=1)
    inserted = mapper.insert_keyframe(second, _frontend_output_for_mapper(1))

    assert inserted == 0
    assert mapper.map.anchor_count() == 1
    assert mapper.stats.last_hash_hits == 1
    assert mapper.stats.last_replace_fused == 0
    assert mapper.stats.last_replace_fused_existing == 0
    assert mapper.stats.last_replace_newly_inserted == 0
    assert int(mapper.map._anchor_last_update_kf_ord[0]) == 0
    assert torch.equal(mapper.map._anchor_grid_coord[0], torch.tensor([1, 0, 0], dtype=torch.int32))


def test_replace_fuse_hash_insert_rejects_new_candidates_in_occupancy_radius():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    seeds = _pfgs_seed_batch(
        torch.tensor([[0.005, 0.0, 0.0], [0.006, 0.0, 0.0], [0.085, 0.0, 0.0]]),
        frame_id=0,
        source_flat_idx=torch.tensor([1, 2, 3], dtype=torch.long),
        source_hw=(1, 4),
    )

    inserted = mapper.insert_keyframe(seeds, _frontend_output_for_mapper(0))

    assert inserted == 2
    assert mapper.map.anchor_count() == 2
    assert mapper.stats.last_dense_seed_candidates == 3
    assert mapper.stats.last_insert_mask_seed_candidates == 3
    assert mapper.stats.last_voxel_seed_candidates == 3
    assert mapper.stats.last_replace_fused_new_duplicate == 1
    assert mapper.stats.last_replace_newly_inserted == 2
    assert all(len(rows) == 1 for rows in mapper._build_replace_fuse_voxel_index().values())


def test_replace_fuse_first_keyframe_ignores_missing_seed_budget():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "first_keyframe_max_seeds": 3,
                "keyframe_max_seeds": 3,
                "global_anchor_budget": 10,
                "max_missing_seeds_per_keyframe": 1,
            }
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    seeds = _pfgs_seed_batch(
        torch.tensor(
            [
                [0.005, 0.0, 0.0],
                [0.055, 0.0, 0.0],
                [0.105, 0.0, 0.0],
            ]
        ),
        frame_id=0,
    )

    inserted = mapper.insert_keyframe(seeds, _frontend_output_for_mapper(0))

    assert inserted == 3
    assert mapper.map.anchor_count() == 3
    assert mapper.stats.last_skipped_missing_budget == 0


def test_replace_fuse_compacts_duplicate_anchors_after_voxel_refresh():
    config = {
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        },
        "BackendOptimization": {"sh_degree": 2},
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    seeds = _pfgs_seed_batch(torch.tensor([[0.005, 0.0, 0.0], [0.055, 0.0, 0.0]]), frame_id=0)
    assert mapper.insert_keyframe(seeds, _frontend_output_for_mapper(0)) == 2
    with torch.no_grad():
        mapper.map.xyz.data[0] = torch.tensor([0.004, 0.0, 0.0])
        mapper.map.xyz.data[1] = torch.tensor([0.016, 0.0, 0.0])
        mapper.map.features.data[:] = mapper.map._inv_sigmoid(
            torch.tensor([[0.2, 0.4, 0.6], [0.6, 0.8, 0.2]], dtype=mapper.map.features.dtype)
        )
        mapper.map.opacity_logit.data[:] = mapper.map._inv_sigmoid(
            torch.tensor([[0.25], [0.75]], dtype=mapper.map.opacity_logit.dtype)
        )
        scales = torch.tensor([[0.02, 0.04, 0.06], [0.06, 0.08, 0.10]], dtype=mapper.map.scaling.dtype)
        mapper.map.scaling.data[:] = torch.log(torch.expm1(scales))
        mapper.map.rotation.data[:] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]],
            dtype=mapper.map.rotation.dtype,
        )
        mapper.map.sh_rest.data[0].fill_(1.0)
        mapper.map.sh_rest.data[1].fill_(3.0)
        mapper.map._anchor_obs_count[:] = torch.tensor([2, 3], dtype=torch.int32)
        mapper.map._anchor_conf_accum[:] = torch.tensor([0.7, 0.2], dtype=torch.float32)
        mapper.map._anchor_last_seen_kf[:] = torch.tensor([5, 7], dtype=torch.int32)
        mapper.map._anchor_last_update_kf_ord[:] = torch.tensor([1, 3], dtype=torch.int32)
        mapper.map._anchor_birth_frame[:] = torch.tensor([10, 8], dtype=torch.int32)
        mapper.map._anchor_inlier_obs[:] = torch.tensor([4, 6], dtype=torch.int32)
        mapper.map._anchor_outlier_obs[:] = torch.tensor([1, 2], dtype=torch.int32)
        mapper.map._anchor_voxel_size[:] = torch.tensor([0.04, 0.06], dtype=torch.float32)

    compacted = mapper._refresh_pfgs360_voxel_cache(compact=True)

    assert compacted == 1
    assert mapper.map.anchor_count() == 1
    assert torch.allclose(mapper.map.get_xyz[0], torch.tensor([0.010, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(mapper.map.get_features[0], torch.tensor([0.4, 0.6, 0.4]), atol=1.0e-6)
    assert torch.allclose(mapper.map.get_opacity[0], torch.tensor([0.5]), atol=1.0e-6)
    assert torch.allclose(mapper.map.get_scaling[0], torch.tensor([0.04, 0.06, 0.08]), atol=3.0e-5)
    assert torch.allclose(mapper.map.get_rotation[0].abs(), torch.tensor([1.0, 0.0, 0.0, 0.0]), atol=1.0e-6)
    assert torch.allclose(mapper.map.sh_rest[0], torch.full_like(mapper.map.sh_rest[0], 2.0), atol=1.0e-6)
    assert int(mapper.map._anchor_obs_count[0]) == 5
    assert torch.allclose(mapper.map._anchor_conf_accum[0], torch.tensor(0.9), atol=1.0e-6)
    assert int(mapper.map._anchor_last_seen_kf[0]) == 7
    assert int(mapper.map._anchor_last_update_kf_ord[0]) == 3
    assert int(mapper.map._anchor_birth_frame[0]) == 8
    assert int(mapper.map._anchor_inlier_obs[0]) == 10
    assert int(mapper.map._anchor_outlier_obs[0]) == 3
    assert torch.allclose(mapper.map._anchor_voxel_size[0], torch.tensor(0.05), atol=1.0e-6)
    index = mapper._build_replace_fuse_voxel_index()
    assert all(len(rows) == 1 for rows in index.values())
