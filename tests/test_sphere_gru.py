import torch
from torch import nn

from frontend.pano_droid.correlation import SphericalCorrBlock, coords_grid
from frontend.pano_droid.encoders import BasicEncoder, ContextEncoder
from frontend.pano_droid.sphere_conv import SphereConv2d
from frontend.pano_droid.sphere_gru import SphereConvGRU


def test_sphere_conv2d_shape_device_dtype_and_backward_cpu():
    conv = SphereConv2d(3, 4, kernel_size=3, bias=True)
    image = torch.randn(2, 3, 8, 16, dtype=torch.float32, requires_grad=True)
    out = conv(image)
    assert out.shape == (2, 4, 8, 16)
    assert out.device == image.device
    assert out.dtype == image.dtype
    out.square().mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert conv.weight.grad is not None
    assert torch.isfinite(conv.weight.grad).all()


def test_sphere_conv2d_cuda_device_dtype_if_available():
    if not torch.cuda.is_available():
        return
    device = torch.device("cuda")
    conv = SphereConv2d(2, 3, kernel_size=3).to(device=device, dtype=torch.float32)
    image = torch.randn(1, 2, 6, 12, device=device, dtype=torch.float32, requires_grad=True)
    out = conv(image)
    assert out.shape == (1, 3, 6, 12)
    assert out.device.type == "cuda"
    assert out.dtype == torch.float32
    out.mean().backward()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()


def test_sphere_conv2d_seam_sampling_stays_finite_and_uses_wrap():
    conv = SphereConv2d(1, 1, kernel_size=3, bias=False)
    nn.init.ones_(conv.conv.weight)
    image = torch.zeros(1, 1, 8, 16)
    image[..., :, -1] = 1.0
    out = conv(image)
    assert torch.isfinite(out).all()
    assert out[..., :, 0].sum() > 0.0


def test_sphere_conv2d_from_conv2d_copies_weights():
    base = nn.Conv2d(2, 3, kernel_size=3, bias=True)
    conv = SphereConv2d.from_conv2d(base)
    assert torch.allclose(conv.weight, base.weight)
    assert conv.bias is not None
    assert torch.allclose(conv.bias, base.bias)


def test_sphere_conv_gru_forward_and_grad():
    gru = SphereConvGRU(hidden_dim=8, input_dim=5, kernel_size=3)
    h = torch.zeros(2, 8, 12, 24, requires_grad=True)
    x = torch.randn(2, 5, 12, 24, requires_grad=True)
    out = gru(h, x)
    assert out.shape == h.shape
    out.mean().backward()
    assert torch.isfinite(x.grad).all()


def test_droid_style_encoders_and_correlation_shapes():
    image = torch.randn(1, 3, 32, 64)
    fnet = BasicEncoder(output_dim=16, base_dim=16)
    cnet = ContextEncoder(hidden_dim=16, context_dim=12, base_dim=16)
    fmap0 = fnet(image)
    fmap1 = fnet(torch.roll(image, shifts=1, dims=-1))
    hidden, context = cnet(image)
    assert fmap0.shape[-2:] == (4, 8)
    assert hidden.shape == (1, 16, 4, 8)
    assert context.shape == (1, 12, 4, 8)
    corr = SphericalCorrBlock(fmap0, fmap1, num_levels=2, radius=1)
    coords = coords_grid(1, 4, 8, device=image.device, dtype=image.dtype)
    out = corr(coords)
    assert out.shape == (1, 18, 4, 8)
    assert torch.isfinite(out).all()


def test_coords_grid_uses_pixel_centers():
    coords = coords_grid(1, 2, 3)
    assert torch.allclose(coords[0, :, 0, 0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(coords[0, :, -1, -1], torch.tensor([2.5, 1.5]))
