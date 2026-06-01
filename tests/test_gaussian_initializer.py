import torch

from frontend.pano_droid.interfaces import FrontendOutput
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

