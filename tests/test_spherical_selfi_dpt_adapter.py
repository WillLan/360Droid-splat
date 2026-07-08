import torch
from torch import nn
import torch.nn.functional as F

from models.panovggt_feature_wrapper import PanoVGGTFeatureWrapper
from models.spherical_selfi_dpt_adapter import SphericalSelfiDPTAdapter


class _Stage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int) -> None:
        super().__init__()
        self.scale = int(scale)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale > 1:
            x = F.avg_pool2d(x, kernel_size=self.scale, stride=self.scale)
        return self.conv(x)


class _DummyPanoVGGT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stage1 = _Stage(3, 4, 1)
        self.stage2 = _Stage(4, 6, 2)
        self.stage3 = _Stage(6, 8, 2)
        self.stage4 = _Stage(8, 10, 2)

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w)
        f1 = self.stage1(flat)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        _ = self.stage4(f3)
        depth = torch.ones(b, v, 1, h, w, device=images.device, dtype=images.dtype)
        poses = torch.eye(4, device=images.device, dtype=images.dtype).view(1, 1, 4, 4).repeat(b, v, 1, 1)
        return {"depth": depth, "poses_c2w": poses}


def test_panovggt_feature_wrapper_freezes_model_and_captures_four_stages():
    model = _DummyPanoVGGT()
    wrapper = PanoVGGTFeatureWrapper(
        model,
        stage_hooks=["stage1", "stage2", "stage3", "stage4"],
    )
    images = torch.rand(1, 2, 3, 32, 64)
    out = wrapper(images)
    assert all(not param.requires_grad for param in model.parameters())
    assert not model.training
    assert len(out.stage_features) == 4
    assert out.feature_shapes == [
        (1, 2, 4, 32, 64),
        (1, 2, 6, 16, 32),
        (1, 2, 8, 8, 16),
        (1, 2, 10, 4, 8),
    ]
    assert out.init_depth is not None and out.init_depth.shape == (1, 2, 1, 32, 64)
    assert out.init_poses is not None and out.init_poses.shape == (1, 2, 4, 4)
    assert all(not feature.requires_grad for feature in out.stage_features)


def test_adapter_default_output_shape_and_l2_norm():
    features = [
        torch.rand(1, 1, 4, 8, 16),
        torch.rand(1, 1, 6, 4, 8),
        torch.rand(1, 1, 8, 2, 4),
        torch.rand(1, 1, 10, 1, 2),
    ]
    adapter = SphericalSelfiDPTAdapter([4, 6, 8, 10], hidden_dim=8)
    out = adapter(features)
    assert out.shape == (1, 1, 24, 504, 1008)
    norm = torch.linalg.norm(out, dim=2)
    assert torch.allclose(norm.mean(), torch.tensor(1.0), atol=1e-5)
    assert norm.min() > 0.999
    assert norm.max() < 1.001


def test_adapter_supports_flattened_bv_features_and_backpropagates_to_adapter_only():
    b, v = 2, 3
    features = [
        torch.rand(b * v, 4, 8, 16),
        torch.rand(b * v, 6, 4, 8),
        torch.rand(b * v, 8, 2, 4),
        torch.rand(b * v, 10, 1, 2),
    ]
    adapter = SphericalSelfiDPTAdapter([4, 6, 8, 10], hidden_dim=8, image_height=32, image_width=64)
    out = adapter(features, batch_size=b, num_views=v)
    assert out.shape == (b, v, 24, 32, 64)
    loss = out[:, :, 0].mean()
    loss.backward()
    grads = [param.grad for param in adapter.parameters() if param.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0 for grad in grads)


def test_adapter_supports_token_features_with_explicit_token_hw():
    b, v = 1, 2
    token_hw = [(8, 16), (4, 8), (2, 4), (1, 2)]
    channels = [4, 6, 8, 10]
    tokens = [
        torch.rand(b, v, h * w, c)
        for (h, w), c in zip(token_hw, channels)
    ]
    adapter = SphericalSelfiDPTAdapter(
        channels,
        hidden_dim=8,
        image_height=32,
        image_width=64,
        token_hw=token_hw,
    )
    out = adapter(tokens)
    assert out.shape == (b, v, 24, 32, 64)
    assert torch.allclose(torch.linalg.norm(out, dim=2), torch.ones(b, v, 32, 64), atol=1e-5)


def test_frozen_wrapper_features_can_train_adapter_without_panovggt_grads():
    model = _DummyPanoVGGT()
    wrapper = PanoVGGTFeatureWrapper(model, stage_hooks=["stage1", "stage2", "stage3", "stage4"])
    adapter = SphericalSelfiDPTAdapter([4, 6, 8, 10], hidden_dim=8, image_height=32, image_width=64)
    features = wrapper(torch.rand(1, 2, 3, 32, 64)).stage_features
    out = adapter(features)
    out[:, :, 0].mean().backward()
    assert all(param.grad is None for param in model.parameters())
    assert any(param.grad is not None and param.grad.abs().sum() > 0 for param in adapter.parameters())
