"""Small CPU Swin2SR models exercise attention, padding, gradients, and checkpoints."""

import pytest
import torch

from sen2sr.models.opensr_baseline.swin import Swin2SR, window_partition, window_reverse


@pytest.fixture(autouse=True)
def reproducible_initialization():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


def small_swin(scale=2):
    return Swin2SR(
        img_size=8,
        in_channels=2,
        out_channels=3,
        embed_dim=8,
        depths=[2],
        num_heads=[2],
        window_size=4,
        upscale=scale,
        upsampler="pixelshuffledirect",
        drop_path_rate=0,
    )


def test_window_partition_round_trip_preserves_pixels():
    image = torch.arange(2 * 8 * 12 * 3).reshape(2, 8, 12, 3)

    windows = window_partition(image, window_size=4)
    restored = window_reverse(windows, window_size=4, H=8, W=12)

    assert windows.shape == (12, 4, 4, 3)
    torch.testing.assert_close(restored, image, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [2, 4])
@pytest.mark.parametrize("height,width", [(8, 8), (7, 9)])
def test_swin_forward_and_backward_with_window_padding(height, width, scale):
    model = small_swin(scale)
    image = torch.randn(1, 2, height, width, requires_grad=True)

    result = model(image)
    result.square().mean().backward()

    assert result.shape == (1, 3, height * scale, width * scale)
    assert torch.isfinite(result).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    attention = model.layers[0].residual_group.blocks[0].attn
    assert attention.qkv.weight.grad is not None
    assert torch.isfinite(attention.qkv.weight.grad).all()


def test_swin_state_dict_round_trip_preserves_predictions():
    model = small_swin().eval()
    restored = small_swin().eval()
    restored.load_state_dict(model.state_dict())
    image = torch.randn(1, 2, 7, 9)

    with torch.no_grad():
        torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)
