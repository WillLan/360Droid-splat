import torch

from frontend.pano_droid.interfaces import FrontendOutput
from frontend.pano_droid.spherical_camera import erp_pixel_to_bearing, pixel_grid
from mapping.gaussian_initializer import GaussianInitializer


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
