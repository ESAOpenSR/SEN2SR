"""Numerical and gradient tests for the CNN architecture, without pretrained weights."""

import pytest
import torch

from sen2sr.models.opensr_baseline.cnn import CNNSR, Conv3XC


@pytest.fixture(autouse=True)
def reproducible_initialization():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        yield


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("relu", [True, False])
def test_folded_convolution_matches_training_path(bias, stride, relu):
    block = Conv3XC(2, 3, gain1=2, s=stride, bias=bias, relu=relu)
    image = torch.randn(2, 2, 7, 9)
    expected = block(image)

    block.train_mode = False
    result = block(image)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_cnn_forward_and_backward_with_rectangular_inputs(scale):
    model = CNNSR(2, 3, feature_channels=4, upscale=scale, num_blocks=2)
    image = torch.randn(2, 2, 7, 9, requires_grad=True)

    result = model(image)
    result.square().mean().backward()

    assert result.shape == (2, 3, 7 * scale, 9 * scale)
    assert torch.isfinite(result).all()
    assert image.grad is not None
    assert torch.isfinite(image.grad).all()
    assert image.grad.abs().sum() > 0
    assert model.upsampler[0].weight.grad is not None
    assert torch.isfinite(model.upsampler[0].weight.grad).all()


def test_cnn_returns_only_requested_attention_maps():
    model = CNNSR(2, 3, feature_channels=4, upscale=2, num_blocks=3).eval()
    image = torch.randn(1, 2, 7, 9)

    with torch.no_grad():
        expected = model(image)
        result, attentions = model(image, save_attentions=[0, 2])

    torch.testing.assert_close(result, expected)
    assert len(attentions) == 2
    for attention in attentions:
        assert attention.shape == (1, 4, 7, 9)
        assert (attention >= -0.5).all()
        assert (attention <= 0.5).all()


def test_cnn_state_dict_round_trip_preserves_predictions():
    model = CNNSR(2, 3, feature_channels=4, upscale=2, num_blocks=2).eval()
    restored = CNNSR(2, 3, feature_channels=4, upscale=2, num_blocks=2).eval()
    restored.load_state_dict(model.state_dict())
    image = torch.randn(1, 2, 7, 9)

    with torch.no_grad():
        torch.testing.assert_close(restored(image), model(image), rtol=0, atol=0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CNNSR intermediate blocks do not feed the following block; changing this affects pretrained outputs",
)
def test_every_cnn_block_contributes_to_prediction():
    model = CNNSR(2, 3, feature_channels=4, upscale=2, num_blocks=3)

    model(torch.randn(1, 2, 7, 9)).square().mean().backward()

    for block in model.blocks:
        gradient = block.c1_r.conv[0].weight.grad
        assert gradient is not None
        assert gradient.abs().sum() > 0
