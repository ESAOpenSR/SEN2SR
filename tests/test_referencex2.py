"""Tests for Sentinel-2 band ordering and the 20m-to-10m wrapper."""

import torch
from torch import nn

from sen2sr.referencex2 import reconstruct_sentinel2_stack, resample_sentinel2_bands, srmodel


def test_resampling_preserves_every_native_resolution_pixel():
    image = torch.arange(2 * 10 * 6 * 8, dtype=torch.float32).reshape(2, 10, 6, 8)
    original = image.clone()

    result = resample_sentinel2_bands(image)

    torch.testing.assert_close(result[:, -4:], image[:, [0, 1, 2, 6]], rtol=0, atol=0)
    torch.testing.assert_close(image, original, rtol=0, atol=0)


def test_resampling_interpolates_20m_bands_with_known_bilinear_values():
    coarse = torch.tensor([[0.0, 2.0], [4.0, 6.0]])
    image = torch.zeros(1, 10, 4, 4)
    slow_bands = [3, 4, 5, 7, 8, 9]
    image[:, slow_bands] = coarse.repeat_interleave(2, -2).repeat_interleave(2, -1)

    result = resample_sentinel2_bands(image)

    expected = torch.tensor([[0, 0.5, 1.5, 2], [1, 1.5, 2.5, 3], [3, 3.5, 4.5, 5], [4, 4.5, 5.5, 6]])
    torch.testing.assert_close(result[:, :6], expected[None, None].expand(1, 6, 4, 4))


def test_resampling_preserves_constant_bands_and_groups_them_by_resolution():
    image = torch.arange(10, dtype=torch.float64).reshape(1, 10, 1, 1).expand(2, 10, 6, 8).clone()
    original = image.clone()

    result = resample_sentinel2_bands(image)

    expected = torch.tensor([3, 4, 5, 7, 8, 9, 0, 1, 2, 6], dtype=image.dtype)
    torch.testing.assert_close(result, expected.reshape(1, 10, 1, 1).expand_as(image))
    torch.testing.assert_close(image, original)


def test_reconstruction_restores_sentinel2_band_order():
    native = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1).expand(2, 4, 3, 5)
    resampled = torch.arange(10, 16, dtype=torch.float32).reshape(1, 6, 1, 1).expand(2, 6, 3, 5)

    result = reconstruct_sentinel2_stack(native, resampled)

    expected = torch.tensor([0, 1, 2, 10, 11, 12, 3, 13, 14, 15], dtype=torch.float32)
    torch.testing.assert_close(result, expected.reshape(1, 10, 1, 1).expand(2, 10, 3, 5))


def test_wrapper_preserves_native_bands_and_refines_resampled_bands():
    class PredictSlowBands(nn.Module):
        def forward(self, image):
            return image[:, :6] - 5

    class Constraint(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.25))

        def forward(self, low_resolution, prediction):
            assert low_resolution.shape[1] == 10
            return prediction + self.offset

    image = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1, 1).expand(2, 10, 6, 8)
    constraint = Constraint()
    model = srmodel(PredictSlowBands(), constraint)

    result = model(image)

    expected = torch.tensor([0, 1, 2, 0.25, 0.25, 0.25, 6, 2.25, 3.25, 4.25])
    torch.testing.assert_close(result, expected.reshape(1, 10, 1, 1).expand_as(image))
    assert not constraint.training
    assert not constraint.offset.requires_grad
