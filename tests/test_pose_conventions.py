from __future__ import annotations

import torch

from frontend.pano_droid.spherical_ba import se3_exp
from geometry.pose import invert_c2w, relative_c2w
from geometry.sim3 import (
    apply_sim3_to_c2w,
    rebase_c2w_to_sim3_anchor,
    sim3_from_components,
)


def test_c2w_inverse_and_relative_pose_map_source_to_target() -> None:
    source = se3_exp(torch.tensor([0.3, -0.2, 0.1, 0.1, -0.05, 0.02]))
    target = se3_exp(torch.tensor([-0.4, 0.1, 0.2, -0.03, 0.08, 0.04]))
    point_source = torch.tensor([0.2, -0.1, 2.0, 1.0])
    point_world = source @ point_source
    expected_target = invert_c2w(target) @ point_world
    actual_target = relative_c2w(source, target) @ point_source
    torch.testing.assert_close(actual_target, expected_target, atol=1.0e-6, rtol=1.0e-6)


def test_sim3_pose_rebase_round_trip_includes_inverse_scale() -> None:
    rotation = se3_exp(torch.tensor([0.0, 0.0, 0.0, 0.1, -0.2, 0.05]))[:3, :3]
    local = se3_exp(torch.tensor([1.2, -0.4, 0.3, -0.05, 0.03, 0.08]))
    for scale in (0.5, 2.0):
        transform = sim3_from_components(
            scale, rotation, torch.tensor([10.0, -2.0, 1.0])
        )
        global_pose = apply_sim3_to_c2w(transform, local)
        recovered = rebase_c2w_to_sim3_anchor(transform, global_pose)
        torch.testing.assert_close(recovered, local, atol=1.0e-5, rtol=1.0e-5)


def test_scale_two_anchor_example_rebases_translation_to_one() -> None:
    transform = sim3_from_components(2.0, torch.eye(3), torch.tensor([10.0, 0.0, 0.0]))
    global_pose = torch.eye(4)
    global_pose[0, 3] = 12.0
    local = rebase_c2w_to_sim3_anchor(transform, global_pose)
    assert float(local[0, 3]) == 1.0
    torch.testing.assert_close(apply_sim3_to_c2w(transform, local), global_pose)
