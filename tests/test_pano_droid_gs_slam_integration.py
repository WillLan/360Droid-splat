import json
import math
from pathlib import Path

import pytest
import torch

from backend.pano_gs import PFGS360Renderer, PanoGaussianMap, PanoGaussianMapper, PanoRenderCamera
from backend.pano_gs.losses import (
    BackendLossWeights,
    backend_render_loss,
    pano_depth_loss,
)
from frontend.pano_droid.interfaces import FrontendOutput
from mapping.gaussian_initializer import GaussianSeedBatch
from models.per_pixel_gaussian_observation import BatchedExplicitPerPixelGaussianSet
from system.pano_droid_gs_slam import PanoDroidGSSlamSystem, _se3_blend_pose


class _CountingRenderer:
    def __init__(self) -> None:
        self.calls = 0
        self.frame_ids: list[int] = []

    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        self.calls += 1
        fid = int(round(float(camera.c2w.detach().cpu()[0, 3]) * 10.0))
        self.frame_ids.append(fid)
        H, W = int(camera.image_height), int(camera.image_width)
        if gaussian_map.get_features.numel() == 0:
            color = torch.zeros(3, device=camera.c2w.device, dtype=camera.c2w.dtype)
        else:
            color = gaussian_map.get_features.mean(dim=0).to(device=camera.c2w.device, dtype=camera.c2w.dtype)
        render = color.view(3, 1, 1).expand(3, H, W)
        return {"render": render, "depth": render.new_ones(1, H, W)}


class _AllGaussianParameterRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        height, width = int(camera.image_height), int(camera.image_width)
        feature_signal = gaussian_map.features.mean(dim=0)
        sh_signal = (
            gaussian_map.sh_rest.mean(dim=(0, 1))
            if gaussian_map.sh_rest.numel() > 0
            else torch.zeros_like(feature_signal)
        )
        opacity_signal = gaussian_map.opacity_logit.mean()
        geometry_signal = (
            gaussian_map.xyz.mean()
            + gaussian_map.scaling.mean()
            + gaussian_map.rotation.mean()
        )
        color = torch.sigmoid(
            feature_signal
            + 0.25 * sh_signal
            + 0.10 * opacity_signal
            + 0.05 * geometry_signal
        )
        render = color.view(3, 1, 1).expand(3, height, width)
        depth_value = 1.0 + torch.nn.functional.softplus(
            gaussian_map.xyz.mean()
            + 0.25 * gaussian_map.scaling.mean()
            + 0.10 * gaussian_map.rotation.mean()
        )
        depth = depth_value.view(1, 1, 1).expand(1, height, width)
        alpha = torch.sigmoid(gaussian_map.opacity_logit.mean()).view(1, 1, 1)
        alpha = alpha.expand(1, height, width)
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "render_distort": None,
        }


class _DepthGateRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        render = torch.zeros(3, H, W, device=camera.c2w.device, dtype=camera.c2w.dtype)
        depth = torch.ones(1, H, W, device=render.device, dtype=render.dtype)
        if W >= 2:
            depth[:, :, 1] = 2.0
        if W >= 3:
            depth[:, :, 2] = 0.6
        alpha = torch.ones(1, H, W, device=render.device, dtype=render.dtype)
        alpha[:, :, 0] = 0.0
        if W >= 5:
            alpha[:, :, 4] = 0.10
        if W >= 6:
            alpha[:, :, 5] = 0.0
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=render.device),
            "radii": torch.zeros(total, dtype=torch.int32, device=render.device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=render.device),
            "render_distort": None,
        }


class _ReplaceBandRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        device = camera.c2w.device
        dtype = camera.c2w.dtype
        render = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth = torch.ones(1, H, W, device=device, dtype=dtype)
        if W >= 2:
            depth[:, :, 1] = 1.15
        if W >= 3:
            depth[:, :, 2] = 1.50
        alpha = torch.ones(1, H, W, device=device, dtype=dtype)
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=device),
            "radii": torch.zeros(total, dtype=torch.int32, device=device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=device),
            "render_distort": None,
        }


class _ReplaceForegroundLargeRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        device = camera.c2w.device
        dtype = camera.c2w.dtype
        render = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth = torch.ones(1, H, W, device=device, dtype=dtype)
        if W >= 2:
            depth[:, :, 1] = 1.30
        if W >= 3:
            depth[:, :, 2] = 0.60
        if W >= 4:
            depth[:, :, 3] = 1.50
        alpha = torch.ones(1, H, W, device=device, dtype=dtype)
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=device),
            "radii": torch.zeros(total, dtype=torch.int32, device=device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=device),
            "render_distort": None,
        }


class _ReplaceMissingDepthRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        device = camera.c2w.device
        dtype = camera.c2w.dtype
        render = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth = torch.ones(1, H, W, device=device, dtype=dtype)
        alpha = torch.ones(1, H, W, device=device, dtype=dtype)
        if W >= 1:
            alpha[:, :, 0] = 0.0
        if W >= 2:
            depth[:, :, 1] = 0.0
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=device),
            "radii": torch.zeros(total, dtype=torch.int32, device=device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=device),
            "render_distort": None,
        }


class _SkyBiasedScaleRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        device = camera.c2w.device
        dtype = camera.c2w.dtype
        render = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth = torch.ones(1, H, W, device=device, dtype=dtype)
        depth[:, :, : max(0, W - 16)] = 4.0
        alpha = torch.ones(1, H, W, device=device, dtype=dtype)
        total = int(gaussian_map.anchor_count())
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.zeros(total, dtype=torch.bool, device=device),
            "radii": torch.zeros(total, dtype=torch.int32, device=device),
            "n_touched": torch.zeros(total, dtype=torch.int32, device=device),
            "render_distort": None,
        }


class _BadEvidenceRenderer:
    def render(self, camera: PanoRenderCamera, gaussian_map: PanoGaussianMap) -> dict:
        H, W = int(camera.image_height), int(camera.image_width)
        device = camera.c2w.device
        dtype = camera.c2w.dtype
        total = int(gaussian_map.anchor_count())
        render = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth = torch.ones(1, H, W, device=device, dtype=dtype)
        alpha = torch.ones(1, H, W, device=device, dtype=dtype)
        return {
            "render": render,
            "depth": depth,
            "alpha": alpha,
            "opacity": alpha,
            "visibility_filter": torch.ones(total, dtype=torch.bool, device=device),
            "radii": torch.ones(total, dtype=torch.int32, device=device),
            "n_touched": torch.ones(total, dtype=torch.int32, device=device),
            "render_distort": None,
        }


def _small_frontend_output(frame_id: int) -> FrontendOutput:
    pose = torch.eye(4)
    pose[0, 3] = float(frame_id) * 0.1
    return FrontendOutput(
        frame_id=frame_id,
        timestamp=float(frame_id),
        pose_c2w=pose,
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=None,
        tracking_status="tracked",
    )


def _frontend_output_with_size(frame_id: int, H: int, W: int) -> FrontendOutput:
    out = _small_frontend_output(frame_id)
    return FrontendOutput(
        frame_id=out.frame_id,
        timestamp=out.timestamp,
        pose_c2w=out.pose_c2w,
        relative_pose=out.relative_pose,
        pose_confidence=out.pose_confidence,
        inverse_depth=torch.ones(1, H, W),
        depth_confidence=torch.ones(1, H, W),
        spherical_flow=out.spherical_flow,
        keyframe_score=out.keyframe_score,
        is_keyframe=out.is_keyframe,
        ba_residual=out.ba_residual,
        tracking_status=out.tracking_status,
    )


def _small_non_keyframe_output(frame_id: int) -> FrontendOutput:
    out = _small_frontend_output(frame_id)
    out.is_keyframe = False
    out.keyframe_score = 0.0
    return out


def _small_seed_batch(frame_id: int) -> GaussianSeedBatch:
    return GaussianSeedBatch(
        xyz=torch.tensor([[0.05 * frame_id, 0.0, 1.0]], dtype=torch.float32),
        rgb=torch.tensor([[0.2 + 0.1 * frame_id, 0.4, 0.7]], dtype=torch.float32),
        confidence=torch.ones(1),
        scale=torch.full((1,), 0.1),
        level=torch.zeros(1, dtype=torch.long),
        frame_id=frame_id,
    )


def _empty_seed_batch(frame_id: int) -> GaussianSeedBatch:
    return GaussianSeedBatch(
        xyz=torch.zeros(0, 3, dtype=torch.float32),
        rgb=torch.zeros(0, 3, dtype=torch.float32),
        confidence=torch.zeros(0, dtype=torch.float32),
        scale=torch.zeros(0, dtype=torch.float32),
        level=torch.zeros(0, dtype=torch.int8),
        frame_id=int(frame_id),
    )


def test_mapper_renders_keyframe_diagnostic():
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(
        gaussian_map,
        renderer=PFGS360Renderer(config=config, allow_fallback=True),
    )
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[-0.1, 0.0, 1.0], [0.1, 0.0, 1.2]], dtype=torch.float32),
        rgb=torch.tensor([[1.0, 0.2, 0.1], [0.1, 0.7, 1.0]], dtype=torch.float32),
        confidence=torch.ones(2),
        scale=torch.full((2,), 0.1),
        level=torch.zeros(2, dtype=torch.long),
        frame_id=7,
    )
    output = FrontendOutput(
        frame_id=7,
        timestamp=7.0,
        pose_c2w=torch.eye(4),
        relative_pose=None,
        pose_confidence=1.0,
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        spherical_flow=None,
        keyframe_score=1.0,
        is_keyframe=True,
        ba_residual=None,
        tracking_status="tracked",
    )
    image = torch.rand(3, 4, 8)

    mapper.insert_keyframe(seeds, output, image=image)
    diagnostic = mapper.render_keyframe_diagnostic(7)

    assert diagnostic is not None
    assert diagnostic.frame_id == 7
    assert diagnostic.target.shape == image.shape
    assert diagnostic.render.shape == image.shape
    assert diagnostic.depth is not None
    assert diagnostic.anchor_count == 2


def test_pfgs360_mapper_render_depth_gate_budgets_missing_and_depth_mismatch_regions():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360",
                "voxel_size": 0.1,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 20,
                "render_alpha_min": 0.2,
                "missing_alpha_min": 0.05,
                "render_depth_rel_threshold": 0.10,
                "max_missing_seeds_per_keyframe": 1,
                "max_depth_mismatch_seeds_per_keyframe": 2,
                "prioritize_depth_mismatch": True,
                "near_grid_radius": 0,
                "reset_after_outlier_observations": 99,
                "prune_after_outlier_observations": 99,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_DepthGateRenderer())
    first = GaussianSeedBatch(
        xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
        rgb=torch.zeros(1, 3),
        confidence=torch.ones(1),
        scale=torch.full((1,), 0.1),
        level=torch.zeros(1, dtype=torch.int8),
        frame_id=0,
    )
    assert mapper.insert_keyframe(first, _small_frontend_output(0), image=torch.zeros(3, 1, 6)) == 1
    seeds = GaussianSeedBatch(
        xyz=torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [3.0, 0.0, 1.0],
                [4.0, 0.0, 1.0],
                [5.0, 0.0, 1.0],
            ]
        ),
        rgb=torch.zeros(6, 3),
        confidence=torch.ones(6),
        scale=torch.full((6,), 0.1),
        level=torch.zeros(6, dtype=torch.int8),
        frame_id=1,
        source_flat_idx=torch.arange(6),
        source_hw=(1, 6),
        insert_enabled=torch.ones(6, dtype=torch.bool),
        insert_score=torch.tensor([1.0, 0.9, 0.8, 0.6, 0.1, 0.7]),
    )

    inserted = mapper.insert_keyframe(seeds, _small_frontend_output(1), image=torch.zeros(3, 1, 6))

    assert inserted == 3
    assert mapper.stats.last_suppressed_insert == 2
    assert mapper.stats.last_render_missing_pixels == 2
    assert mapper.stats.last_render_depth_mismatch_pixels == 2
    assert mapper.stats.last_render_bad_pixels == 4
    assert mapper.stats.last_missing_seed_candidates == 2
    assert mapper.stats.last_depth_mismatch_seed_candidates == 2
    assert mapper.stats.last_skipped_missing_budget == 1
    assert mapper.stats.last_skipped_depth_mismatch_budget == 0
    diagnostic = mapper.last_depth_insertion_diagnostic
    assert diagnostic is not None
    assert diagnostic.render_depth is not None
    assert diagnostic.predicted_depth is not None
    assert diagnostic.rel_depth_error is not None
    assert diagnostic.render_bad_mask is not None
    assert int(diagnostic.render_bad_mask.sum()) == 4
    assert torch.equal(mapper.last_inserted_source_flat_idx, torch.tensor([0, 1, 2]))


def test_replace_fuse_inserts_missing_or_depth_error_above_delete_min():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "replace_insert_rel_min": 0.10,
                "replace_delete_rel_min": 0.10,
                "replace_delete_rel_max": 0.20,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceBandRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _small_frontend_output(0),
        image=torch.zeros(3, 1, 4),
    )

    masks, stats = mapper._pfgs360_replace_fuse_masks_and_delete(
        _small_frontend_output(1),
        torch.zeros(3, 1, 4),
        sky_mask=None,
    )

    assert masks is not None
    assert torch.equal(masks["insert"], torch.tensor([[[False, True, True, False]]]))
    assert torch.equal(masks["delete"], torch.tensor([[[False, True, False, False]]]))
    assert stats["replace_deleted"] == 0


def test_replace_fuse_deletes_large_error_only_when_render_depth_is_closer():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "replace_insert_rel_min": 0.20,
                "replace_delete_rel_min": 0.20,
                "replace_delete_rel_max": 0.30,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceForegroundLargeRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _small_frontend_output(0),
        image=torch.zeros(3, 1, 4),
    )

    masks, stats = mapper._pfgs360_replace_fuse_masks_and_delete(
        _small_frontend_output(1),
        torch.zeros(3, 1, 4),
        sky_mask=None,
    )

    assert masks is not None
    assert torch.equal(masks["insert"], torch.tensor([[[False, True, True, True]]]))
    assert torch.equal(masks["delete"], torch.tensor([[[False, True, True, False]]]))
    assert stats["depth_mismatch_pixels"] == 2
    assert stats["replace_deleted"] == 0


def test_replace_fuse_deletes_foreground_and_near_back_depth_anchors():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "insert_occupancy_radius_voxels": 0.0,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "replace_front_depth_abs_tol": 0.03,
                "replace_front_depth_rel_tol": 0.02,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceBandRenderer())
    seeds = GaussianSeedBatch(
        xyz=torch.tensor(
            [
                [0.80, 0.0, 0.0],
                [1.02, 0.0, 0.0],
                [1.05, 0.0, 0.0],
                [1.20, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        rgb=torch.zeros(4, 3),
        confidence=torch.ones(4),
        scale=torch.full((4,), 0.01),
        level=torch.zeros(4, dtype=torch.int8),
        frame_id=0,
    )
    assert mapper.insert_keyframe(seeds, _small_frontend_output(0), image=torch.zeros(3, 1, 4)) == 4

    deleted = mapper._delete_responsible_replace_fuse_anchors(
        torch.ones(1, 1, 4, dtype=torch.bool),
        torch.ones(1, 1, 4, dtype=torch.float32),
        {"visibility_filter": torch.ones(4, dtype=torch.bool)},
        _small_frontend_output(0),
        1,
        4,
    )

    assert deleted == 2
    xyz = mapper.map.get_xyz.detach().cpu()
    assert torch.allclose(torch.sort(xyz[:, 0]).values, torch.tensor([1.05, 1.20]), atol=1.0e-6)


def test_replace_fuse_non_first_inserts_from_pred_depth_mask_without_initializer_seeds():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "replace_insert_rel_min": 0.10,
                "replace_delete_rel_min": 0.10,
                "replace_delete_rel_max": 0.20,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 20,
                "gaussian_scale_mode": "erp_depth_latitude",
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceBandRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _frontend_output_with_size(0, 1, 4),
        image=torch.zeros(3, 1, 4),
    )

    inserted = mapper.insert_keyframe(
        _empty_seed_batch(1),
        _frontend_output_with_size(1, 1, 4),
        image=torch.zeros(3, 1, 4),
    )

    assert inserted == 2
    assert mapper.stats.last_insert_mask_pixels == 2
    assert mapper.stats.last_pred_depth_generated_seeds == 2
    assert mapper.stats.last_pred_depth_invalid_pixels == 0
    assert mapper.stats.last_dense_seed_candidates == 2
    assert mapper.stats.last_insert_mask_seed_candidates == 2
    assert mapper.stats.last_voxel_seed_candidates == 2
    assert set(mapper.last_inserted_source_flat_idx.tolist()) == {1, 2}


def test_replace_fuse_pred_depth_generation_respects_sky_mask():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "replace_insert_rel_min": 0.10,
                "replace_delete_rel_min": 0.10,
                "replace_delete_rel_max": 0.20,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 20,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceBandRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _frontend_output_with_size(0, 1, 4),
        image=torch.zeros(3, 1, 4),
    )
    sky_mask = torch.tensor([[[False, True, False, False]]])

    inserted = mapper.insert_keyframe(
        _empty_seed_batch(1),
        _frontend_output_with_size(1, 1, 4),
        image=torch.zeros(3, 1, 4),
        sky_mask=sky_mask,
    )

    assert inserted == 1
    assert mapper.stats.last_insert_mask_pixels == 1
    assert mapper.stats.last_pred_depth_generated_seeds == 1
    assert mapper.last_inserted_source_flat_idx.tolist() == [2]


def test_replace_fuse_missing_ignores_low_alpha_when_render_depth_is_valid():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "missing_alpha_min": 0.95,
                "replace_insert_rel_min": 0.10,
                "replace_delete_rel_min": 0.10,
                "replace_delete_rel_max": 0.20,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_ReplaceMissingDepthRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _small_frontend_output(0),
        image=torch.zeros(3, 1, 3),
    )

    masks, stats = mapper._pfgs360_replace_fuse_masks_and_delete(
        _small_frontend_output(1),
        torch.zeros(3, 1, 3),
        sky_mask=None,
    )

    assert masks is not None
    assert torch.equal(masks["missing"], torch.tensor([[[False, True, False]]]))
    assert torch.equal(masks["insert"], torch.tensor([[[False, True, False]]]))
    assert stats["missing_pixels"] == 1


def test_replace_fuse_depth_scale_shift_excludes_sky_pixels():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "render_alpha_min": 0.20,
                "replace_insert_rel_min": 0.10,
                "replace_delete_rel_min": 0.10,
                "replace_delete_rel_max": 0.20,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
            }
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_SkyBiasedScaleRenderer())
    mapper.insert_keyframe(
        GaussianSeedBatch(
            xyz=torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32),
            rgb=torch.zeros(1, 3),
            confidence=torch.ones(1),
            scale=torch.full((1,), 0.02),
            level=torch.zeros(1, dtype=torch.int8),
            frame_id=0,
        ),
        _frontend_output_with_size(0, 1, 40),
        image=torch.zeros(3, 1, 40),
    )
    sky_mask = torch.zeros(1, 1, 40, dtype=torch.bool)
    sky_mask[:, :, :24] = True

    masks, stats = mapper._pfgs360_replace_fuse_masks_and_delete(
        _frontend_output_with_size(1, 1, 40),
        torch.zeros(3, 1, 40),
        sky_mask=sky_mask,
    )

    assert masks is not None
    assert not bool(masks["insert"].any())
    assert not bool(masks["delete"].any())
    assert stats["render_bad_pixels"] == 0
    diagnostic = mapper.last_depth_insertion_diagnostic
    assert diagnostic is not None
    assert diagnostic.depth_scale == pytest.approx(1.0)


def test_mapper_force_sky_render_uses_skybox_inside_sky_mask():
    config = {
        "SkyBox": {
            "enabled": True,
            "force_sky_render": True,
            "optimization_mask_enable": True,
        }
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    gs = torch.zeros(3, 1, 2)
    gs[0] = 1.0
    sky = torch.zeros(3, 1, 2)
    sky[2] = 1.0
    pkg = {
        "render": gs,
        "gs_only": gs,
        "sky_bg_only": sky,
        "sky_bg_alpha": torch.ones(1, 1, 2),
        "alpha": torch.ones(1, 1, 2),
    }

    out = mapper._apply_skybox_optimization_mask(pkg, torch.tensor([[[True, False]]]))

    assert bool(out["skybox_force_sky_render"])
    assert torch.allclose(out["render"][:, 0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(out["render"][:, 0, 1], torch.tensor([1.0, 0.0, 0.0]))


def test_backend_l1_dssim_loss_has_gradient_without_opacity_penalty():
    render = torch.zeros(3, 4, 8, requires_grad=True)
    target = torch.ones(3, 4, 8)
    alpha = torch.ones(1, 4, 8, requires_grad=True)
    weights = BackendLossWeights(
        photometric_mode="l1_dssim",
        rgb_l1_weight=0.8,
        dssim_weight=0.2,
        depth=0.0,
        opacity=0.0,
    )

    loss, metrics = backend_render_loss({"render": render, "alpha": alpha}, target, weights=weights)
    loss.backward()

    assert torch.isfinite(loss)
    assert render.grad is not None
    assert float(render.grad.abs().sum()) > 0.0
    assert alpha.grad is None or float(alpha.grad.abs().sum()) == 0.0
    assert float(metrics["opacity"]) > 0.0


def test_backend_render_loss_masks_sky_pixels_for_rgb_and_depth():
    render = torch.zeros(3, 4, 8, requires_grad=True)
    target = torch.ones(3, 4, 8)
    render_depth = torch.ones(1, 4, 8, requires_grad=True)
    target_depth = torch.full((1, 4, 8), 2.0)
    non_sky = torch.zeros(1, 4, 8, dtype=torch.bool)
    non_sky[:, :, :4] = True
    weights = BackendLossWeights(
        photometric_mode="l1_dssim",
        rgb_l1_weight=0.8,
        dssim_weight=0.2,
        depth=0.03,
        opacity=0.0,
    )

    loss, metrics = backend_render_loss(
        {"render": render, "depth": render_depth},
        target,
        target_depth=target_depth,
        photometric_mask=non_sky,
        depth_mask=non_sky,
        weights=weights,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert float(metrics["photometric"]) > 0.0
    assert float(metrics["depth"]) > 0.0
    assert float(render.grad[:, :, :4].abs().sum()) > 0.0
    assert float(render.grad[:, :, 4:].abs().max()) == 0.0
    assert float(render_depth.grad[:, :, :4].abs().sum()) > 0.0
    assert float(render_depth.grad[:, :, 4:].abs().max()) == 0.0


def test_log_depth_huber_is_confidence_weighted_and_ignores_invalid_depth() -> None:
    render_depth = torch.tensor(
        [[[1.0, 2.0, 4.0, float("nan")]]], requires_grad=True
    )
    target_depth = torch.tensor([[[1.0, 1.0, 0.0, 2.0]]])
    confidence = torch.tensor([[[1.0, 0.5, 1.0, 1.0]]])

    loss = pano_depth_loss(
        render_depth,
        target_depth,
        confidence=confidence,
        mode="log_huber",
        huber_delta=0.1,
    )
    loss.backward()

    expected = 0.1 * (math.log(2.0) - 0.05)
    assert float(loss.detach()) == pytest.approx(expected / 3.0, rel=1.0e-5)
    assert torch.isfinite(render_depth.grad).all()
    assert float(render_depth.grad[0, 0, 0]) == 0.0
    assert float(render_depth.grad[0, 0, 2]) == 0.0
    assert float(render_depth.grad[0, 0, 3]) == 0.0


def test_mapper_observation_depth_confidence_excludes_sky_mask():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {"sky_mask_source": "panovggt_head"},
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"))
    image = torch.zeros(3, 4, 8)
    inverse_depth = torch.ones(1, 4, 8)
    confidence = torch.ones(1, 4, 8)
    sky_mask = torch.zeros(1, 4, 8, dtype=torch.bool)
    sky_mask[:, :2, :] = True

    mapper.register_observation_values(
        frame_id=7,
        image=image,
        c2w=torch.eye(4),
        inverse_depth=inverse_depth,
        depth_confidence=confidence,
        is_keyframe=False,
        sky_mask=sky_mask,
    )

    obs = mapper.observations[7]
    assert obs.depth_confidence is not None
    assert float(obs.depth_confidence[:, :2, :].sum()) == 0.0
    assert float(obs.depth_confidence[:, 2:, :].min()) == 1.0


def test_mapper_random_window_optimizes_one_sample_per_step():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 5,
            "window_keyframes": 3,
            "random_window_frame_per_iter": True,
            "sample_keyframes_per_step": 1,
            "pose_window_keyframes": 3,
            "fixed_window_frames": 1,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in range(3):
        image = torch.full((3, 4, 8), 0.25 + 0.1 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)

    metrics = mapper.optimize_after_keyframe()

    assert renderer.calls == 5
    assert metrics["window_size"] == 3.0
    assert metrics["sampled_window_size"] == 1.0
    assert metrics["trainable_pose_count"] == 2.0
    assert mapper.stats.last_phase == "sliding_window"
    assert mapper.stats.last_window_size == 3
    assert len(mapper.stats.last_sampled_keyframes) == 1
    assert mapper.stats.last_trainable_pose_count == 2


def test_mapper_frontend_graph_window_prioritizes_history_hints():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 1,
            "window_keyframes": 2,
            "use_frontend_graph_window": True,
            "random_window_frame_per_iter": False,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in (0, 4, 8, 12):
        image = torch.full((3, 4, 8), 0.2 + 0.01 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)
    mapper.set_frontend_graph_window_ids([4, 12])

    metrics = mapper.optimize_after_keyframe()

    assert metrics["window_size"] == 3.0
    assert mapper.stats.last_window_keyframes == [4, 8, 12]
    assert renderer.frame_ids == [4, 8, 12]


def test_mapper_feedforward_window_uses_history_and_non_keyframes():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "sliding_window_steps": 1,
            "random_window_frame_per_iter": False,
            "final_global_steps": 0,
            "optimize_skybox": False,
            "FeedForwardWindow": {
                "enabled": True,
                "history_keyframes": 2,
                "optimize_non_keyframe_observations": True,
                "gaussian_scope": "selected_birth_keyframes",
                "prune": {"enabled": False},
            },
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in (1, 2, 3):
        image = torch.full((3, 4, 8), 0.2 + 0.01 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)
    for frame_id in (10, 11):
        image = torch.full((3, 4, 8), 0.1 + 0.01 * frame_id, dtype=torch.float32)
        mapper.register_observation(_small_non_keyframe_output(frame_id), image)

    metrics = mapper.optimize_feedforward_window(current_frame_ids=[10, 11], history_frame_ids=[1, 2, 3])

    assert metrics["window_size"] == 4.0
    assert mapper.stats.last_window_observations == [2, 3, 10, 11]
    assert mapper.stats.last_window_keyframes == [2, 3]
    assert renderer.frame_ids == [2, 3, 10, 11]


def test_spherical_selfi_window_runs_twenty_balanced_steps_and_reuses_overlap_pose():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "pose_prior_weight": 0.0,
            "final_global_steps": 0,
            "optimize_skybox": False,
            "FeedForwardWindow": {"enabled": True, "prune": {"enabled": False}},
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)
    mapper.insert_keyframe(
        _small_seed_batch(0),
        _small_frontend_output(0),
        image=torch.full((3, 4, 8), 0.2),
    )
    gaussian_map._anchor_owner_window_id[:] = 0
    for frame_id in (1, 2, 3):
        mapper.register_observation(
            _small_frontend_output(frame_id),
            torch.full((3, 4, 8), 0.2 + 0.01 * frame_id),
            is_keyframe=True,
        )
    assert mapper.prepare_spherical_selfi_window((0, 1, 2, 3)) == 4
    overlap_pose_delta = mapper.pose_deltas[3]
    metrics = mapper.optimize_spherical_selfi_window(
        window_id=0,
        frame_ids=(0, 1, 2, 3),
        iters=20,
        settings={
            "fixed_pose_frame_ids": [0],
            "pose_grad_clip": 1.0e-3,
            "sampler_seed": 17,
        },
    )
    assert metrics["steps"] == 20.0
    assert renderer.calls == 20
    assert [metrics[f"sample_count_frame_{frame_id}"] for frame_id in range(4)] == [5.0] * 4

    for frame_id in (4, 5, 6):
        mapper.register_observation(
            _small_frontend_output(frame_id),
            torch.full((3, 4, 8), 0.25),
            is_keyframe=True,
        )
    assert mapper.prepare_spherical_selfi_window((3, 4, 5, 6)) == 4
    assert mapper.pose_deltas[3] is overlap_pose_delta


def test_gaussian_only_staged_updates_all_gaussian_groups_but_keeps_pose_and_owner() -> None:
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "sh_degree": 2,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "optimize_skybox": False,
            "FeedForwardWindow": {"enabled": True},
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(
        gaussian_map, renderer=_AllGaussianParameterRenderer()
    )
    mapper.insert_keyframe(
        _small_seed_batch(0),
        _small_frontend_output(0),
        image=torch.full((3, 4, 8), 0.85),
    )
    gaussian_map._anchor_owner_window_id[:] = 0
    gaussian_map.configure_lazy_owner_transforms(True)
    gaussian_map.set_lazy_owner_transform(0, torch.eye(4), set_reference=True)
    gaussian_map.set_lazy_owner_transform(0, torch.eye(4))
    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    parameter_before = {
        name: value.detach().clone()
        for name, value in gaussian_map.named_parameters()
    }
    pose_before = mapper.pose_deltas[0].delta.detach().clone()
    owner_before = gaussian_map._lazy_owner_current_transforms[0].clone()

    metrics = mapper.optimize_spherical_selfi_staged(
        window_id=0,
        frame_ids=(0,),
        active_owner_window_ids=(0,),
        settings={
            "sample_observations_per_step": 2,
            "appearance": {"steps": 2},
            "geometry": {"steps": 2},
            "acceptance": {
                "appearance_min_improvement": -1.0,
                "geometry_min_improvement": -1.0,
                "geometry_max_rgb_worsening": 1.0,
            },
        },
    )

    assert metrics["appearance_accepted"] == 1.0, metrics
    assert metrics["geometry_accepted"] == 1.0
    assert metrics["pose_unchanged"] == 1.0
    assert metrics["owner_transform_unchanged"] == 1.0
    for name in (
        "features",
        "sh_rest",
        "opacity_logit",
        "xyz",
        "scaling",
        "rotation",
    ):
        assert not torch.equal(
            dict(gaussian_map.named_parameters())[name], parameter_before[name]
        ), name
    assert torch.equal(mapper.pose_deltas[0].delta, pose_before)
    assert torch.equal(
        gaussian_map._lazy_owner_current_transforms[0], owner_before
    )


def test_gaussian_only_joint_3dgs_updates_every_group_without_parameter_clamps() -> None:
    config = {
        "MapRepresentation": {
            "mode": "anchor_scaffold_panorama",
            "gaussian_parameterization": "traditional_3dgs",
        },
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "sh_degree": 2,
            "gaussian_refine_enable": True,
            "optimize_skybox": False,
            "FeedForwardWindow": {"enabled": True},
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(
        gaussian_map, renderer=_AllGaussianParameterRenderer()
    )
    gaussian_map.add_seeds(
        GaussianSeedBatch(
            xyz=torch.tensor(
                [
                    [-1.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                ]
            ),
            rgb=torch.full((4, 3), 0.2),
            confidence=torch.ones(4),
            scale=torch.full((4,), 0.1),
            level=torch.zeros(4, dtype=torch.long),
            frame_id=0,
        )
    )
    gaussian_map._anchor_owner_window_id[:] = torch.tensor([-1, 0, 1, 2])
    gaussian_map.scaling.data.fill_(25.0)
    gaussian_map.opacity_logit.data.fill_(15.0)
    gaussian_map.rotation.data.copy_(
        torch.tensor([[2.0, 1.0, 0.5, -0.25]]).expand(4, -1)
    )
    mapper.register_observation(
        _small_frontend_output(0),
        torch.full((3, 4, 8), 0.85),
        is_keyframe=True,
    )
    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    before = {
        name: value.detach().clone()
        for name, value in gaussian_map.named_parameters()
    }
    pose_before = mapper.pose_deltas[0].delta.detach().clone()

    metrics = mapper.optimize_spherical_selfi_joint_3dgs(
        window_id=2,
        frame_ids=(0,),
        active_owner_window_ids=(0, 1, 2),
        settings={"steps": 2, "max_rgb_worsening": 100.0},
    )

    assert metrics["accepted"] == 1.0, metrics
    assert metrics["pose_unchanged"] == 1.0
    assert metrics["owner_transform_unchanged"] == 1.0
    for name in (
        "features",
        "sh_rest",
        "opacity_logit",
        "xyz",
        "scaling",
        "rotation",
    ):
        value = dict(gaussian_map.named_parameters())[name]
        torch.testing.assert_close(value[0], before[name][0])
        assert not torch.equal(value[1:], before[name][1:]), name
    assert float(gaussian_map.scaling[1:].min()) > 20.0
    assert float(gaussian_map.opacity_logit[1:].min()) > 12.0
    assert not torch.allclose(
        torch.linalg.norm(gaussian_map.rotation[1:], dim=-1),
        torch.ones(3),
    )
    assert torch.equal(mapper.pose_deltas[0].delta, pose_before)


def test_gaussian_only_joint_3dgs_rolls_back_rgb_regression(monkeypatch) -> None:
    config = {
        "MapRepresentation": {"gaussian_parameterization": "traditional_3dgs"},
        "BackendOptimization": {
            "enabled": True,
            "sh_degree": 2,
            "FeedForwardWindow": {"enabled": True},
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(
        gaussian_map, renderer=_AllGaussianParameterRenderer()
    )
    gaussian_map.add_seeds(_small_seed_batch(0))
    gaussian_map._anchor_owner_window_id[:] = 0
    mapper.register_observation(
        _small_frontend_output(0), torch.full((3, 4, 8), 0.5), is_keyframe=True
    )
    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    before = {
        name: value.detach().clone()
        for name, value in gaussian_map.named_parameters()
    }
    evaluations = iter(
        (
            {"loss": 1.0, "photometric": 1.0, "finite": 1.0},
            {"loss": 1.01, "photometric": 1.01, "finite": 1.0},
        )
    )
    monkeypatch.setattr(
        mapper,
        "evaluate_spherical_selfi_window",
        lambda *args, **kwargs: next(evaluations),
    )

    metrics = mapper.optimize_spherical_selfi_joint_3dgs(
        window_id=0,
        frame_ids=(0,),
        active_owner_window_ids=(0,),
        settings={"steps": 1, "max_rgb_worsening": 0.005},
    )

    assert metrics["accepted"] == 0.0
    assert metrics["window_rollback"] == 1.0
    for name, value in gaussian_map.named_parameters():
        torch.testing.assert_close(value, before[name])


def test_staged_geometry_rejection_keeps_accepted_appearance(monkeypatch) -> None:
    config = {
        "BackendOptimization": {
            "enabled": True,
            "sh_degree": 2,
            "gaussian_refine_enable": True,
            "FeedForwardWindow": {"enabled": True},
        }
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_CountingRenderer())
    mapper.insert_keyframe(
        _small_seed_batch(0),
        _small_frontend_output(0),
        image=torch.full((3, 4, 8), 0.5),
    )
    gaussian_map._anchor_owner_window_id[:] = 0
    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    feature_before = gaussian_map.features.detach().clone()
    xyz_before = gaussian_map.xyz.detach().clone()
    evaluations = iter(
        (
            {"loss": 1.0, "photometric": 1.0, "finite": 1.0},
            {"loss": 0.5, "photometric": 0.5, "finite": 1.0},
            {"loss": 0.5, "photometric": 0.5, "finite": 1.0},
            {"loss": 0.7, "photometric": 0.7, "finite": 1.0},
        )
    )
    calls = 0

    def fake_evaluate(*args, **kwargs):
        return next(evaluations)

    def fake_optimize(**kwargs):
        nonlocal calls
        mapper._spherical_selfi_rollback_state = (
            {
                name: value.detach().clone()
                for name, value in gaussian_map.named_parameters()
            },
            {
                frame_id: pose.delta.detach().clone()
                for frame_id, pose in mapper.pose_deltas.items()
            },
        )
        with torch.no_grad():
            if calls == 0:
                gaussian_map.features.add_(0.1)
            else:
                gaussian_map.xyz.add_(1.0)
        calls += 1
        return {"steps": 1.0, "non_finite_window": 0.0}

    monkeypatch.setattr(mapper, "evaluate_spherical_selfi_window", fake_evaluate)
    monkeypatch.setattr(mapper, "optimize_spherical_selfi_window", fake_optimize)

    metrics = mapper.optimize_spherical_selfi_staged(
        window_id=0,
        frame_ids=(0,),
        active_owner_window_ids=(0,),
    )

    assert metrics["appearance_accepted"] == 1.0
    assert metrics["geometry_accepted"] == 0.0
    assert metrics["geometry_rollback"] == 1.0
    assert not torch.equal(gaussian_map.features, feature_before)
    assert torch.equal(gaussian_map.xyz, xyz_before)


def test_staged_appearance_rejection_restores_the_full_window(monkeypatch) -> None:
    config = {
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "FeedForwardWindow": {"enabled": True},
        }
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_CountingRenderer())
    mapper.insert_keyframe(
        _small_seed_batch(0),
        _small_frontend_output(0),
        image=torch.full((3, 4, 8), 0.5),
    )
    gaussian_map._anchor_owner_window_id[:] = 0
    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    parameter_before = {
        name: value.detach().clone()
        for name, value in gaussian_map.named_parameters()
    }
    evaluations = iter(
        (
            {"loss": 1.0, "photometric": 1.0, "finite": 1.0},
            {"loss": 1.0, "photometric": 1.0, "finite": 1.0},
        )
    )

    monkeypatch.setattr(
        mapper,
        "evaluate_spherical_selfi_window",
        lambda *args, **kwargs: next(evaluations),
    )

    def fake_optimize(**kwargs):
        with torch.no_grad():
            gaussian_map.features.add_(0.2)
            gaussian_map.xyz.add_(0.2)
        return {"steps": 1.0, "non_finite_window": 0.0}

    monkeypatch.setattr(mapper, "optimize_spherical_selfi_window", fake_optimize)

    metrics = mapper.optimize_spherical_selfi_staged(
        window_id=0,
        frame_ids=(0,),
        active_owner_window_ids=(0,),
    )

    assert metrics["appearance_accepted"] == 0.0
    assert metrics["window_rollback"] == 1.0
    for name, value in gaussian_map.named_parameters():
        assert torch.equal(value, parameter_before[name])


def test_spherical_selfi_window_initializes_cubemap_from_head_sky_mask():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {
            "enabled": True,
            "resolution": 4,
            "optimize": True,
            "force_sky_render": True,
            "init_fallback_to_full_image": False,
        },
        "BackendOptimization": {
            "enabled": True,
            "FeedForwardWindow": {"enabled": True},
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_CountingRenderer())
    sky_mask = torch.zeros(1, 4, 8, dtype=torch.bool)
    sky_mask[:, :2] = True
    image = torch.zeros(3, 4, 8)
    image[:, :2] = torch.tensor([0.2, 0.4, 0.8]).view(3, 1, 1)
    mapper.register_observation(
        _small_frontend_output(0),
        image,
        is_keyframe=True,
        sky_mask=sky_mask,
    )

    assert mapper.prepare_spherical_selfi_window((0,)) == 1
    assert gaussian_map._skybox_initialized
    assert gaussian_map.get_skybox_faces is not None
    assert float(gaussian_map.get_skybox_faces.detach().mean()) > 0.0


def test_replace_fuse_chunk_optimizer_samples_current_chunk_and_recent_keyframes():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
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
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": True,
            "steps_per_chunk": 3,
            "sample_frames_per_step": 1,
            "current_chunk_observation_frames": 4,
            "recent_keyframe_observation_frames": 2,
            "recent_insert_keyframes": 2,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in (1, 2):
        image = torch.full((3, 4, 8), 0.2 + 0.01 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)
    for frame_id in (10, 11, 12, 13):
        image = torch.full((3, 4, 8), 0.1 + 0.01 * frame_id, dtype=torch.float32)
        mapper.register_observation(_small_non_keyframe_output(frame_id), image)
    renderer.calls = 0
    renderer.frame_ids.clear()

    metrics = mapper.optimize_feedforward_window(current_frame_ids=[10, 11, 12, 13], history_frame_ids=[99])

    assert renderer.calls == 3
    assert metrics["window_size"] == 6.0
    assert metrics["sampled_window_size"] == 1.0
    assert mapper.stats.last_window_observations == [10, 11, 12, 13, 1, 2]
    assert len(mapper.stats.last_sampled_keyframes) == 1


def test_replace_fuse_chunk_optimizer_uses_recent_chunk_and_keyframe_window():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
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
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": True,
            "steps_per_chunk": 1,
            "sample_frames_per_step": 1,
            "current_chunk_observation_frames": 4,
            "recent_chunk_observation_chunks": 4,
            "recent_keyframe_observation_frames": 4,
            "recent_insert_keyframes": 4,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    for frame_id in (1, 2, 3, 4):
        image = torch.full((3, 4, 8), 0.2 + 0.01 * frame_id, dtype=torch.float32)
        mapper.insert_keyframe(_small_seed_batch(frame_id), _small_frontend_output(frame_id), image=image)
    for frame_id in range(10, 30):
        image = torch.full((3, 4, 8), 0.1 + 0.01 * frame_id, dtype=torch.float32)
        mapper.register_observation(_small_non_keyframe_output(frame_id), image)

    metrics = mapper.optimize_feedforward_window(current_frame_ids=list(range(10, 30)), history_frame_ids=[])

    assert metrics["window_size"] == 20.0
    assert mapper.stats.last_window_observations == [*range(14, 30), 1, 2, 3, 4]
    assert mapper.stats.last_window_keyframes == [1, 2, 3, 4]
    assert metrics["sampled_window_size"] == 1.0


def test_replace_fuse_chunk_optimizer_limits_gaussian_updates_to_active_keyframe_scope():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "insert_occupancy_radius_voxels": 0.0,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "compact_voxels": False,
            }
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": True,
            "steps_per_chunk": 1,
            "sample_frames_per_step": 3,
            "current_chunk_observation_frames": 4,
            "recent_keyframe_observation_frames": 0,
            "final_global_steps": 0,
            "optimize_skybox": False,
            "feature_lr": 1.0e-2,
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_CountingRenderer())
    for ord_idx, frame_id in enumerate((1, 2, 3)):
        start = mapper.map.anchor_count()
        mapper.map.add_seeds(
            _small_seed_batch(frame_id),
            voxel_size=0.02,
            last_update_kf_ord=ord_idx,
        )
        end = mapper.map.anchor_count()
        out = _small_frontend_output(frame_id)
        image = torch.zeros(3, 4, 8)
        mapper._register_keyframe(out, image, start=start, end=end)
        mapper.register_observation(out, image, is_keyframe=True)
    mapper.stats.n_keyframes = len(mapper.keyframes)

    before = mapper.map.features.detach().clone()
    metrics = mapper.optimize_feedforward_window(
        current_frame_ids=[1, 2, 3],
        history_frame_ids=[],
        active_keyframe_ids=[2],
    )
    after = mapper.map.features.detach()

    changed = (after - before).abs().sum(dim=1) > 0.0
    assert changed.tolist() == [False, True, False]
    assert mapper.stats.last_window_keyframes == [1, 2, 3]
    assert mapper.stats.last_active_keyframes == [2]
    assert metrics["active_keyframe_count"] == 1.0
    assert metrics["active_gaussian_count"] == 1.0


def test_replace_fuse_recent_update_mask_tracks_recent_four_keyframes():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
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
        "BackendOptimization": {
            "enabled": True,
            "optimize_after_every_chunk": True,
            "recent_insert_keyframes": 4,
        },
    }
    mapper = PanoGaussianMapper(PanoGaussianMap(config=config, device="cpu"), renderer=_CountingRenderer())
    mapper.map.add_seeds(
        GaussianSeedBatch(
            xyz=torch.tensor([[float(idx), 0.0, 1.0] for idx in range(5)], dtype=torch.float32),
            rgb=torch.full((5, 3), 0.5, dtype=torch.float32),
            confidence=torch.ones(5),
            scale=torch.full((5,), 0.1),
            level=torch.zeros(5, dtype=torch.long),
            frame_id=0,
        ),
        voxel_size=0.02,
        last_update_kf_ord=0,
    )
    mapper.map._anchor_last_update_kf_ord = torch.arange(5, dtype=torch.int32)
    mapper.stats.n_keyframes = 5

    active = mapper._active_anchor_mask_for_recent_updates().detach().cpu()
    assert active.tolist() == [False, True, True, True, True]


def test_replace_fuse_chunk_optimizer_prunes_visible_sky_gaussians():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
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
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": True,
            "steps_per_chunk": 1,
            "sample_frames_per_step": 1,
            "recent_keyframe_observation_frames": 0,
            "recent_insert_keyframes": 2,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_BadEvidenceRenderer())
    mapper.insert_keyframe(_small_seed_batch(1), _small_frontend_output(1), image=torch.zeros(3, 4, 8))
    mapper.register_observation_values(
        frame_id=10,
        image=torch.zeros(3, 4, 8),
        c2w=_small_non_keyframe_output(10).pose_c2w,
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        is_keyframe=False,
        sky_mask=torch.ones(1, 4, 8, dtype=torch.bool),
    )

    metrics = mapper.optimize_feedforward_window(current_frame_ids=[10], history_frame_ids=[])

    assert metrics["sky_pruned"] == 1.0
    assert metrics["profile_backend_feedforward_window_sky_pruned"] == 1.0
    assert mapper.stats.last_sky_pruned == 1
    assert gaussian_map.anchor_count() == 0


def test_replace_fuse_chunk_optimizer_can_disable_sky_prune():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "voxel_size": 0.02,
                "first_keyframe_max_seeds": 10,
                "keyframe_max_seeds": 10,
                "global_anchor_budget": 10,
                "sky_prune_enabled": False,
            }
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": True,
            "steps_per_chunk": 1,
            "sample_frames_per_step": 1,
            "recent_keyframe_observation_frames": 0,
            "recent_insert_keyframes": 2,
            "final_global_steps": 0,
            "optimize_skybox": False,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_BadEvidenceRenderer())
    mapper.insert_keyframe(_small_seed_batch(1), _small_frontend_output(1), image=torch.zeros(3, 4, 8))
    mapper.register_observation_values(
        frame_id=10,
        image=torch.zeros(3, 4, 8),
        c2w=_small_non_keyframe_output(10).pose_c2w,
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        is_keyframe=False,
        sky_mask=torch.ones(1, 4, 8, dtype=torch.bool),
    )

    metrics = mapper.optimize_feedforward_window(current_frame_ids=[10], history_frame_ids=[])

    assert metrics["sky_pruned"] == 0.0
    assert metrics["profile_backend_feedforward_window_sky_pruned"] == 0.0
    assert mapper.stats.last_sky_pruned == 0
    assert gaussian_map.anchor_count() == 1


def test_mapper_feedforward_window_freezes_non_window_gaussians():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "sliding_window_steps": 1,
            "random_window_frame_per_iter": False,
            "final_global_steps": 0,
            "optimize_skybox": False,
            "gaussian_lr": 0.1,
            "FeedForwardWindow": {
                "enabled": True,
                "history_keyframes": 0,
                "gaussian_scope": "selected_birth_keyframes",
                "prune": {"enabled": False},
            },
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = _CountingRenderer()
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    mapper.insert_keyframe(_small_seed_batch(1), _small_frontend_output(1), image=torch.zeros(3, 4, 8))
    mapper.insert_keyframe(_small_seed_batch(99), _small_frontend_output(99), image=torch.zeros(3, 4, 8))
    before = gaussian_map.features.detach().clone()

    mapper.optimize_feedforward_window(current_frame_ids=[1], history_frame_ids=[])

    after = gaussian_map.features.detach()
    assert not torch.allclose(after[0], before[0])
    assert torch.allclose(after[1], before[1])


def test_mapper_feedforward_prune_only_removes_active_bad_gaussians():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "sliding_window_steps": 1,
            "random_window_frame_per_iter": False,
            "final_global_steps": 0,
            "optimize_skybox": False,
            "FeedForwardWindow": {
                "enabled": True,
                "history_keyframes": 0,
                "gaussian_scope": "selected_birth_keyframes",
                "prune": {
                    "enabled": True,
                    "reset_after_bad": 1,
                    "prune_after_bad": 2,
                    "min_seen": 1,
                    "min_bad_ratio": 0.5,
                    "max_inlier_count": 0,
                    "opacity_after_reset": 0.01,
                    "max_prune_per_window": 10,
                    "photo_error_threshold": 0.0,
                },
            },
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    mapper = PanoGaussianMapper(gaussian_map, renderer=_BadEvidenceRenderer())

    mapper.insert_keyframe(_small_seed_batch(1), _small_frontend_output(1), image=torch.ones(3, 4, 8))
    mapper.insert_keyframe(_small_seed_batch(99), _small_frontend_output(99), image=torch.ones(3, 4, 8))

    first = mapper.optimize_feedforward_window(current_frame_ids=[1], history_frame_ids=[])
    assert first["feedforward_opacity_resets"] == 1.0
    assert first["feedforward_pruned"] == 0.0
    assert gaussian_map.anchor_count() == 2

    second = mapper.optimize_feedforward_window(current_frame_ids=[1], history_frame_ids=[])
    assert second["feedforward_pruned"] == 1.0
    assert gaussian_map.anchor_count() == 1
    assert gaussian_map._anchor_birth_frame.tolist() == [99]


def test_pano_gaussian_map_saves_legacy_3dgs_ply_schema(tmp_path: Path):
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    gaussian_map.add_seeds(_small_seed_batch(0))
    ply_path = tmp_path / "point_cloud.ply"

    gaussian_map.save_ply(ply_path)

    data = ply_path.read_bytes()
    header = data.split(b"end_header")[0].decode("ascii") + "end_header"
    expected = [
        "ply",
        "format binary_little_endian 1.0",
        "element vertex 1",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        *[f"property float f_rest_{idx}" for idx in range(24)],
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
    ]
    assert header.splitlines() == expected
    assert b"red" not in data.split(b"end_header")[0]
    assert ply_path.stat().st_size > len(header)


def test_pfgs360_renderer_converts_rgb_to_sh_dc_for_rasterizer():
    config = {"Training": {"panorama_render_mode": "pfgs360_gsplat"}}
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32),
        rgb=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]], dtype=torch.float32),
        confidence=torch.ones(2),
        scale=torch.full((2,), 0.1),
        level=torch.zeros(2, dtype=torch.long),
        frame_id=0,
    )
    gaussian_map.add_seeds(seeds)
    seen: dict[str, torch.Tensor | tuple[int, ...] | int] = {}

    def fake_rasterization(**kwargs):
        colors = kwargs["colors"]
        seen["colors_shape"] = tuple(colors.shape)
        seen["colors"] = colors.detach().clone()
        seen["sh_degree"] = int(kwargs["sh_degree"])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        device = colors.device
        dtype = colors.dtype
        render = torch.zeros(1, height, width, 4, device=device, dtype=dtype)
        rgb = colors[:, 0, :] * 0.28209479177387814 + 0.5
        render[0, :, :, :3] = rgb.mean(dim=0).view(1, 1, 3)
        render[0, :, :, 3] = 1.0
        alpha = torch.ones(1, height, width, 1, device=device, dtype=dtype)
        info = {
            "means2d": torch.zeros(1, colors.shape[0], 2, device=device, dtype=dtype),
            "radii": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
            "accum_times": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
        }
        return render, alpha, None, info

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(image_height=4, image_width=8, c2w=torch.eye(4))
    pkg = renderer._render_gsplat360(fake_rasterization, camera, gaussian_map, torch.zeros(3))

    expected_sh = ((gaussian_map.get_features.detach() - 0.5) / 0.28209479177387814).unsqueeze(1)
    assert seen["colors_shape"] == (2, 1, 3)
    assert torch.allclose(seen["colors"], expected_sh)
    assert seen["sh_degree"] == gaussian_map.active_sh_degree
    assert torch.allclose(pkg["render"], gaussian_map.get_features.mean(dim=0).view(3, 1, 1).expand(3, 4, 8))


def test_pfgs360_renderer_forwards_query_attribution_outputs():
    gaussian_map = PanoGaussianMap(config={}, device="cpu")
    gaussian_map.add_seeds(_small_seed_batch(0))
    seen = {}

    def fake_rasterization(**kwargs):
        seen.update(kwargs)
        height, width = int(kwargs["height"]), int(kwargs["width"])
        render = torch.zeros(1, height, width, 4)
        alpha = torch.ones(1, height, width, 1)
        info = {
            "means2d": torch.zeros(1, 1, 2),
            "radii": torch.ones(1, 1, dtype=torch.int32),
            "accum_times": torch.ones(1, 1, dtype=torch.int32),
            "accum_visible": torch.tensor([[0.75]]),
            "query_answers": torch.tensor([[[0.25, 0.50]]]),
        }
        return render, alpha, None, info

    query = torch.zeros(4, 8, 2)
    query[:, 0, 0] = 1.0
    query[:, -1, 1] = 1.0
    renderer = PFGS360Renderer(config={}, allow_fallback=True)
    package = renderer._render_gsplat360(
        fake_rasterization,
        PanoRenderCamera(4, 8, torch.eye(4)),
        gaussian_map,
        torch.zeros(3),
        query_values=query,
    )

    assert tuple(seen["query_values"].shape) == (1, 4, 8, 2)
    assert seen["query_values"][0, 0, 0, 0] == 1.0
    assert seen["query_values"][0, 0, -1, 1] == 1.0
    torch.testing.assert_close(package["accum_visible"], torch.tensor([0.75]))
    torch.testing.assert_close(
        package["query_answers"], torch.tensor([[0.25, 0.50]])
    )


def test_cpu_erp_fallback_keeps_front_and_back_hemispheres():
    class _Gaussians:
        get_xyz = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        get_features = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        get_opacity = torch.ones(2, 1)

    renderer = PFGS360Renderer(config={}, allow_fallback=True)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))
    pkg = renderer._render_fallback(camera, _Gaussians(), torch.zeros(3))
    assert int(pkg["visibility_filter"].sum()) == 2
    assert int((pkg["alpha"] > 0).sum()) == 2


def test_pfgs360_renderer_batches_four_cameras_in_one_unpacked_call():
    camera_count, gaussian_count = 4, 3
    config = {
        "Training": {
            "pfgs360_packed": False,
            "pfgs360_render_mode": "RGB+ED",
            "pfgs360_rasterize_mode": "antialiased",
        }
    }
    explicit = BatchedExplicitPerPixelGaussianSet(
        xyz=torch.randn(gaussian_count, 3),
        scaling=torch.full((gaussian_count, 3), 0.1),
        rotation=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(gaussian_count, -1).clone(),
        opacity=torch.full((gaussian_count, 1), 0.2),
        features=torch.rand(camera_count, gaussian_count, 3, dtype=torch.bfloat16),
        confidence=torch.full((gaussian_count, 1), 0.2),
        source_frame_index=torch.arange(gaussian_count),
        source_pixel_uv=torch.zeros(gaussian_count, 2),
        source_ray=torch.tensor([[0.0, 0.0, 1.0]]).expand(gaussian_count, -1).clone(),
        source_depth=torch.ones(gaussian_count, 1),
        config=config,
    )
    cameras = [PanoRenderCamera(4, 8, torch.eye(4)) for _ in range(camera_count)]
    calls: list[dict] = []

    def fake_rasterization(**kwargs):
        calls.append(kwargs)
        height, width = int(kwargs["height"]), int(kwargs["width"])
        render = torch.zeros(camera_count, height, width, 4)
        alpha = torch.ones(camera_count, height, width, 1)
        info = {
            "means2d": torch.zeros(camera_count, gaussian_count, 2),
            "radii": torch.ones(camera_count, gaussian_count, dtype=torch.int32),
            "accum_times": torch.ones(camera_count, gaussian_count, dtype=torch.int32),
            "accum_visible": torch.tensor(
                [[1.0, 0.0, 1.0]] * camera_count,
                dtype=torch.float32,
            ),
        }
        return render, alpha, None, info

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    package = renderer._render_gsplat360_cameras(
        fake_rasterization,
        cameras,
        explicit,
        torch.zeros(camera_count, 3),
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["packed"] is False
    assert call["viewmats"].shape == (camera_count, 4, 4)
    assert call["Ks"].shape == (camera_count, 3, 3)
    assert call["colors"].shape == (camera_count, gaussian_count, 1, 3)
    assert call["colors"].dtype == torch.float32
    assert call["opacities"].shape == (gaussian_count,)
    assert package["render"].shape == (camera_count, 3, 4, 8)
    assert package["visibility_filter"].shape == (camera_count, gaussian_count)
    assert torch.equal(
        package["visibility_filter"],
        torch.tensor([[True, False, True]] * camera_count),
    )


def test_pfgs360_renderer_uses_configured_sh_degree_two_coefficients():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendOptimization": {"sh_degree": 2},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    seeds = GaussianSeedBatch(
        xyz=torch.tensor([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]], dtype=torch.float32),
        rgb=torch.tensor([[0.9, 0.2, 0.1], [0.1, 0.8, 0.3]], dtype=torch.float32),
        confidence=torch.ones(2),
        scale=torch.full((2,), 0.1),
        level=torch.zeros(2, dtype=torch.long),
        frame_id=0,
    )
    gaussian_map.add_seeds(seeds)
    with torch.no_grad():
        gaussian_map.sh_rest.fill_(0.125)
    seen: dict[str, torch.Tensor | tuple[int, ...] | int] = {}

    def fake_rasterization(**kwargs):
        colors = kwargs["colors"]
        seen["colors_shape"] = tuple(colors.shape)
        seen["colors"] = colors.detach().clone()
        seen["sh_degree"] = int(kwargs["sh_degree"])
        height = int(kwargs["height"])
        width = int(kwargs["width"])
        device = colors.device
        dtype = colors.dtype
        render = torch.zeros(1, height, width, 4, device=device, dtype=dtype)
        alpha = torch.ones(1, height, width, 1, device=device, dtype=dtype)
        info = {
            "means2d": torch.zeros(1, colors.shape[0], 2, device=device, dtype=dtype),
            "radii": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
            "accum_times": torch.ones(1, colors.shape[0], device=device, dtype=torch.int32),
        }
        return render, alpha, None, info

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(image_height=4, image_width=8, c2w=torch.eye(4))
    renderer._render_gsplat360(fake_rasterization, camera, gaussian_map, torch.zeros(3))

    expected = gaussian_map.get_sh_coefficients.detach()
    assert gaussian_map.active_sh_degree == 2
    assert seen["colors_shape"] == (2, 9, 3)
    assert torch.allclose(seen["colors"], expected)
    assert seen["sh_degree"] == 2


def test_backend_feedback_se3_blend_and_hard_gate():
    source = torch.eye(4)
    target = torch.eye(4)
    target[0, 3] = 2.0
    blended = _se3_blend_pose(source, target, 0.5)
    assert torch.allclose(blended[:3, 3], torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)

    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 8, "width": 16},
        "Frontend": {"mode": "panovggt_long"},
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [8, 16],
            "chunk_size": 2,
            "overlap": 1,
            "emit_delay": 0,
        },
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "BackendFeedback": {
            "enabled": True,
            "blend_alpha": 1.0,
            "reject_first_keyframe_pose_feedback": True,
            "log_decisions": True,
        },
        "Renderer": {"allow_smoke_fallback": True},
    }
    system = PanoDroidGSSlamSystem(cfg)
    assert hasattr(system.frontend, "pose_by_frame")

    for frame_id in range(2):
        system.mapper.insert_keyframe(
            _small_seed_batch(frame_id),
            _small_frontend_output(frame_id),
            image=torch.full((3, 4, 8), 0.3 + 0.1 * frame_id),
        )
        system.frontend.pose_by_frame[frame_id] = _small_frontend_output(frame_id).pose_c2w

    with torch.no_grad():
        system.mapper.pose_deltas[1].delta[0] = 1.0

    updates, decisions = system._collect_backend_feedback_updates({"steps": 1.0, "loss": 0.1})
    by_id = {int(item["frame_id"]): item for item in decisions}

    assert by_id[0]["accepted"] is False
    assert by_id[0]["reason"] == "first_keyframe_rejected"
    assert by_id[1]["accepted"] is True
    assert set(updates) == {1}
    assert torch.allclose(updates[1][0, 3], torch.tensor(1.1), atol=1e-5)
    assert system._apply_backend_feedback_updates(updates) == 1
    assert torch.allclose(system.frontend.pose_by_frame[1].cpu(), updates[1], atol=1e-5)


def test_skybox_renders_without_anchors():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {"enabled": True, "resolution": 8, "optimize": True},
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    image = torch.zeros(3, 8, 16)
    image[1] = 0.35
    image[2] = 1.0
    assert gaussian_map.initialize_skybox_from_image(image, torch.eye(4))

    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))
    pkg = renderer.render(camera, gaussian_map)

    assert pkg["render"].shape == image.shape
    assert float(pkg["render"][2].detach().mean()) > 0.5
    assert float(pkg["sky_bg_alpha"].detach().mean()) > 0.9


def test_skybox_optimization_mask_blocks_non_sky_gradients():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
            "optimization_mask_enable": True,
            "sky_mask_top_ratio": 0.5,
            "sky_mask_min_blue": 0.4,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    renderer = PFGS360Renderer(config=config, allow_fallback=True)
    mapper = PanoGaussianMapper(gaussian_map, renderer=renderer)

    image = torch.zeros(3, 8, 16)
    image[0, 4:, :] = 1.0
    image[1, :4, :] = 0.35
    image[2, :4, :] = 1.0
    camera = PanoRenderCamera(image_height=8, image_width=16, c2w=torch.eye(4))
    pkg = renderer.render(camera, gaussian_map)
    sky_rgb = pkg["sky_bg_only"]
    assert torch.is_tensor(sky_rgb) and sky_rgb.requires_grad
    sky_rgb.retain_grad()

    sky_mask = mapper._skybox_mask_for_target(image)
    masked_pkg = mapper._apply_skybox_optimization_mask(pkg, sky_mask)
    loss, _ = backend_render_loss(masked_pkg, image)
    loss.backward()

    assert torch.allclose(masked_pkg["render"][:, 4:, :], masked_pkg["gs_only"][:, 4:, :])
    assert float(masked_pkg["render"][:, :4, :].detach().abs().sum()) > 0.0
    grad = sky_rgb.grad
    assert torch.is_tensor(grad)
    assert float(grad[:, :4, :].abs().sum()) > 0.0
    assert float(grad[:, 4:, :].abs().max()) == 0.0


def test_skybox_init_requires_sky_mask_by_default():
    config = {
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "SkyBox": {
            "enabled": True,
            "resolution": 8,
            "optimize": True,
            "sky_mask_top_ratio": 0.5,
            "sky_mask_min_blue": 0.4,
        },
    }
    gaussian_map = PanoGaussianMap(config=config, device="cpu")
    image = torch.zeros(3, 8, 16)
    image[0] = 1.0

    assert not gaussian_map.initialize_skybox_from_image(image, torch.eye(4))
    assert gaussian_map._skybox_initialized is False


def test_mapper_panovggt_sky_mask_source_requires_explicit_mask():
    cfg = {
        "Mapping": {"sky_mask_source": "panovggt_head"},
        "SkyBox": {
            "enabled": True,
            "optimization_mask_enable": True,
            "sky_mask_top_ratio": 1.0,
            "sky_mask_min_blue": 0.4,
        },
        "Renderer": {"allow_smoke_fallback": True},
    }
    gaussian_map = PanoGaussianMap(config=cfg)
    mapper = PanoGaussianMapper(gaussian_map, renderer=_CountingRenderer())
    image = torch.zeros(3, 4, 8)
    image[2] = 1.0

    with pytest.raises(ValueError, match="requires explicit sky_mask"):
        mapper.register_observation_values(
            frame_id=3,
            image=image,
            c2w=torch.eye(4),
            inverse_depth=torch.ones(1, 4, 8),
            depth_confidence=torch.ones(1, 4, 8),
        )

    explicit = torch.zeros(1, 4, 8, dtype=torch.bool)
    mapper.register_observation_values(
        frame_id=3,
        image=image,
        c2w=torch.eye(4),
        inverse_depth=torch.ones(1, 4, 8),
        depth_confidence=torch.ones(1, 4, 8),
        sky_mask=explicit,
    )

    assert int(mapper.observations[3].sky_mask.sum()) == 0


def test_system_runs_synthetic_smoke(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 3, "height": 16, "width": 32},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 20,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=3)
    assert summary["frames"] == 3
    assert summary["keyframes"] >= 1
    assert summary["anchors"] > 0
    assert (tmp_path / "summary.json").is_file()
    assert summary["keyframe_decisions_path"] is None


def test_local_ba_diagnostics_are_saved_when_global_backend_is_disabled(
    tmp_path: Path,
    monkeypatch,
):
    class DiagnosticFrontend:
        def __init__(self) -> None:
            self.pending = [
                {
                    "window_id": 0,
                    "frame_ids": (0, 1, 2, 3),
                    "matcher": "adapter",
                    "accepted": True,
                    "num_factors": 512,
                    "initial_median_residual_deg": 1.5,
                    "final_median_residual_deg": 0.5,
                    "matching_sec": 0.1,
                    "ba_sec": 0.2,
                    "ba_diagnostics": {
                        "reason": "accepted",
                        "accepted_steps": 2,
                        "initial_objective": 4.0,
                        "final_objective": 1.0,
                        "published_pose_updated": True,
                        "published_pose_twist_norms": [0.0, 0.01, 0.02, 0.03],
                        "published_translation_update_norms": [0.0, 0.005, 0.01, 0.015],
                        "published_rotation_update_deg": [0.0, 0.1, 0.2, 0.3],
                    },
                }
            ]

        def initialize(self, sequence_meta) -> None:
            del sequence_meta

        def track(self, frame) -> FrontendOutput:
            return _small_frontend_output(int(frame.frame_id))

        def consume_local_ba_diagnostics(self):
            records, self.pending = self.pending, []
            return records

    monkeypatch.setattr(
        "system.pano_droid_gs_slam.build_frontend_from_config",
        lambda config: DiagnosticFrontend(),
    )
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 4, "width": 8},
        "Frontend": {"mode": "spherical_selfi_window"},
        "SphericalSelfiGlobalBackend": {"enabled": False},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Mapping": {
            "max_seeds_per_keyframe": 4,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": False},
        "Results": {"save_dir": str(tmp_path)},
    }
    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=1)
    assert summary["local_ba"] == {
        "windows": 1,
        "accepted": 1,
        "accepted_ratio": 1.0,
        "mean_valid_factors": 512.0,
        "mean_matching_sec": 0.1,
        "mean_ba_sec": 0.2,
    }
    records = json.loads((tmp_path / "local_ba_windows.json").read_text(encoding="utf-8"))
    assert records[0]["published_pose_updated"] is True
    assert records[0]["published_pose_twist_norms"][-1] == 0.03


def test_system_first_chunk_block_keyframe_keeps_all_init_frames(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 4, "height": 8, "width": 16},
        "Frontend": {"mode": "panovggt_long", "keyframe_threshold": 2.0, "force_keyframe_interval": 99},
        "PanoVGGT": {
            "engine": "fake",
            "image_size": [8, 16],
            "chunk_size": 4,
            "overlap": 2,
            "emit_delay": 0,
            "align_mode": "sim3",
            "min_overlap_points": 4,
            "M3Sphere": {"enabled": True},
            "KeyframeAnchor": {
                "enabled": True,
                "prepend_previous_keyframe": False,
                "min_keyframe_interval": 0,
                "max_keyframe_interval": 0,
            },
            "DenseBA": {"enabled": False},
        },
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "seed_source": "world_points_only",
            "max_seeds_per_keyframe": 0,
            "min_depth_confidence": 0.0,
            "sky_mask_enable": False,
            "refine_steps_per_keyframe": 0,
            "NovelGaussianInsertion": {
                "enabled": True,
                "strategy": "pfgs360_replace_fuse",
                "insert_keyframe_policy": "new_block_last",
                "insert_keyframe_block_size": 4,
                "first_chunk_multiframe_init": True,
                "voxel_size": 0.02,
                "insert_occupancy_radius_voxels": 0.0,
                "first_keyframe_max_seeds": 0,
                "keyframe_max_seeds": 0,
                "max_missing_seeds_per_keyframe": 0,
                "max_depth_mismatch_seeds_per_keyframe": 0,
            },
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": False,
            "pose_refine_enable": False,
            "optimize_after_every_chunk": False,
            "keyframe_steps": 0,
            "non_keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 0,
            "final_global_steps": 0,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": False},
        "Results": {"save_dir": str(tmp_path)},
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=4)

    assert summary["keyframes"] == 1
    assert any("first chunk initialization used frames [0, 1, 2, 3]" in note for note in summary["notes"])


def test_system_saves_keyframe_optimized_render_and_depth(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 3, "height": 12, "width": 24},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 12,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": True,
            "local_submap_steps": 1,
            "local_window_keyframes": 2,
            "sliding_window_steps": 0,
            "final_global_steps": 0,
            "fixed_window_frames": 1,
        },
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": True, "save_kf_opt": True, "kf_opt_log_every": 1},
        "Results": {"save_dir": str(tmp_path), "kf_render_format": "png"},
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=3)

    assert summary["anchors"] > 0
    assert summary["backend_optimization_steps"] > 0
    assert any((tmp_path / "kf_renders_opt").glob("kf_*.png"))
    assert any((tmp_path / "kf_depths_opt").glob("kf_*.png"))


def test_system_saves_final_artifacts_and_skybox(tmp_path: Path):
    cfg = {
        "Dataset": {"synthetic": True, "synthetic_length": 1, "height": 10, "width": 20},
        "Frontend": {"keyframe_threshold": 0.0, "force_keyframe_interval": 1},
        "Model": {"feature_dim": 16, "context_dim": 16, "hidden_dim": 16, "update_iters": 1},
        "MapRepresentation": {"mode": "anchor_scaffold_panorama"},
        "Training": {"panorama_render_mode": "pfgs360_gsplat"},
        "Hierarchical": {"voxel_size_lis": [0.2, 0.6, 1.8]},
        "Mapping": {
            "max_seeds_per_keyframe": 8,
            "min_depth_confidence": 0.0,
            "refine_steps_per_keyframe": 0,
            "BootstrapOptimization": {"enabled": True, "first_keyframe_steps": 1, "save_every": 1},
        },
        "BackendOptimization": {
            "enabled": True,
            "gaussian_refine_enable": True,
            "pose_refine_enable": False,
            "keyframe_steps": 0,
            "non_keyframe_steps": 0,
            "local_submap_steps": 0,
            "sliding_window_steps": 0,
            "final_global_steps": 0,
            "optimize_skybox": True,
        },
        "SkyBox": {"enabled": True, "resolution": 8, "optimize": True},
        "Renderer": {"allow_smoke_fallback": True},
        "WeightsAndBiases": {"mode": "disabled"},
        "Visualization": {"save_local": True, "save_kf_opt": True},
        "Results": {
            "save_dir": str(tmp_path),
            "kf_render_format": "png",
            "save_final_ply": True,
            "save_final_checkpoint": True,
            "save_final_keyframe_renders": True,
            "render_final_all_frames": True,
            "save_skybox_previews": True,
            "skybox_preview_height": 16,
            "skybox_preview_width": 32,
        },
    }

    summary = PanoDroidGSSlamSystem(cfg).run(max_frames=2)

    assert summary["artifacts"]["final_ply"]
    assert (tmp_path / "point_cloud" / "init" / "point_cloud.ply").is_file()
    assert any((tmp_path / "point_cloud" / "init").glob("frame_*.ply"))
    assert Path(summary["artifacts"]["final_ply"]).is_file()
    assert Path(summary["artifacts"]["final_checkpoint"]).is_file()
    checkpoint = torch.load(summary["artifacts"]["final_checkpoint"], map_location="cpu", weights_only=False)
    assert "anchor_birth_frame" in checkpoint
    assert "anchor_outlier_obs" in checkpoint
    assert summary["artifacts"]["final_keyframe_render_count"] >= 1
    assert summary["artifacts"]["final_all_frames"]["metrics"]["render_count"] >= 1
    assert summary["final_all_frames_mean_psnr"] is not None
    assert (tmp_path / "final_all_frames" / "metrics.json").is_file()
    assert Path(summary["artifacts"]["final_skybox_erp_preview"]).is_file()
    assert Path(summary["artifacts"]["final_skybox_faces"]).is_file()
    assert any((tmp_path / "init_vis").rglob("iter_*_render.png"))
