import torch

from geometry.spherical_erp import erp_pixel_to_unit_ray
from geometry.spherical_pseudo_correspondence import SphericalCorrespondence
from losses.spherical_selfi_alignment_loss import SphericalSelfiAlignmentLoss


def _corr_from_uv(uv: torch.Tensor, *, height: int, width: int, target_shift: float = 0.0) -> SphericalCorrespondence:
    tgt_uv = uv.clone()
    tgt_uv[:, 0] = torch.remainder(tgt_uv[:, 0] + float(target_shift), float(width))
    tgt_ray = erp_pixel_to_unit_ray(tgt_uv, height, width)
    valid = torch.ones(1, uv.shape[0], dtype=torch.bool)
    return SphericalCorrespondence(
        src_view=torch.zeros(1, uv.shape[0], dtype=torch.long),
        tgt_view=torch.ones(1, uv.shape[0], dtype=torch.long),
        src_uv=uv.unsqueeze(0),
        tgt_uv=uv.unsqueeze(0),
        src_ray=erp_pixel_to_unit_ray(uv, height, width).unsqueeze(0),
        tgt_ray=tgt_ray.unsqueeze(0),
        valid_mask=valid,
        visibility=valid.clone(),
        weight=torch.ones(1, uv.shape[0]),
    )


def test_spherical_selfi_alignment_loss_backpropagates():
    height, width = 8, 16
    features = torch.randn(1, 2, 24, height, width, requires_grad=True)
    uv = torch.tensor([[4.5, 3.5], [8.5, 4.5], [12.5, 5.5]], dtype=torch.float32)
    corr = _corr_from_uv(uv, height=height, width=width)
    loss_fn = SphericalSelfiAlignmentLoss(
        mode="global_lowres",
        loss_stride=2,
        temperature=0.2,
        max_queries=3,
        erp_aux_weight=0.0,
    )
    loss, metrics = loss_fn(features, corr)
    assert torch.isfinite(loss)
    assert metrics["num_queries"].item() == 3
    assert "median_angular_deg" in metrics
    assert "pck_1deg" in metrics
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0


def test_local_fullres_loss_is_smaller_when_predicted_ray_equals_target_ray():
    height, width = 8, 16
    features = torch.randn(1, 2, 24, height, width)
    uv = torch.tensor([[4.5, 3.5], [8.5, 4.5]], dtype=torch.float32)
    loss_fn = SphericalSelfiAlignmentLoss(
        mode="local_fullres",
        local_window_radius=0,
        temperature=0.07,
        max_queries=2,
        erp_aux_weight=0.0,
    )
    same, _ = loss_fn(features, _corr_from_uv(uv, height=height, width=width, target_shift=0.0))
    shifted, _ = loss_fn(features, _corr_from_uv(uv, height=height, width=width, target_shift=4.0))
    assert same < 1.0e-4
    assert shifted > same + 0.5


def test_empty_correspondence_returns_zero_loss():
    features = torch.randn(1, 2, 24, 8, 16, requires_grad=True)
    uv = torch.tensor([[4.5, 3.5]], dtype=torch.float32)
    corr = _corr_from_uv(uv, height=8, width=16)
    corr.valid_mask[:] = False
    corr.weight[:] = 0.0
    loss, metrics = SphericalSelfiAlignmentLoss(max_queries=1)(features, corr)
    assert loss.item() == 0.0
    assert metrics["num_queries"].item() == 0.0
    loss.backward()
    assert features.grad is not None
    assert features.grad.abs().sum() == 0.0


def test_alignment_loss_can_return_predicted_matches_for_visualization():
    height, width = 8, 16
    features = torch.randn(1, 2, 24, height, width)
    uv = torch.tensor([[4.5, 3.5], [8.5, 4.5]], dtype=torch.float32)
    loss_fn = SphericalSelfiAlignmentLoss(
        mode="global_lowres",
        loss_stride=2,
        temperature=0.2,
        max_queries=2,
        erp_aux_weight=0.0,
    )
    loss, metrics, matches = loss_fn(features, _corr_from_uv(uv, height=height, width=width), return_matches=True)
    assert torch.isfinite(loss)
    assert metrics["num_queries"].item() == 2
    assert matches["src_uv"].shape == (2, 2)
    assert matches["tgt_uv"].shape == (2, 2)
    assert matches["pred_uv"].shape == (2, 2)
    assert matches["flat_src"].shape == (2,)
    assert matches["flat_tgt"].shape == (2,)


def test_global_loss_handles_many_queries_per_view_without_expanding_maps():
    height, width = 12, 24
    features = torch.randn(1, 2, 8, height, width, requires_grad=True)
    yy, xx = torch.meshgrid(
        torch.arange(0.5, float(height), 1.0),
        torch.arange(0.5, float(width), 1.0),
        indexing="ij",
    )
    uv = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    corr = _corr_from_uv(uv, height=height, width=width)
    loss_fn = SphericalSelfiAlignmentLoss(
        mode="global_lowres",
        loss_stride=2,
        temperature=0.2,
        max_queries=None,
        erp_aux_weight=0.0,
    )

    loss, metrics = loss_fn(features, corr)

    assert torch.isfinite(loss)
    assert metrics["num_queries"].item() == uv.shape[0]
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
