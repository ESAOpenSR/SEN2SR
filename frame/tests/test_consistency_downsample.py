"""Tests for frame.consistency.downsample -- the deterministic method used to
reduce an SR output back onto its parent LR pixel grid for self-consistency
comparison (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).
"""

import torch
import pytest

from frame.consistency.downsample import AREA_AVERAGE_POOL, downsample_to_lr_grid
from frame.consistency.errors import ScaleFactorError, ShapeMismatchError


def test_constant_image_downsamples_to_the_same_constant():
    sr = torch.full((1, 8, 8), 3.0)
    out = downsample_to_lr_grid(sr, scale_factor=4)
    assert out.shape == (1, 2, 2)
    assert torch.allclose(out, torch.full((1, 2, 2), 3.0))


def test_each_output_pixel_is_the_mean_of_its_block():
    # A single 4x4 block (scale_factor=4, one LR pixel) with known values.
    sr = torch.tensor([[[
        1.0, 1.0, 3.0, 3.0,
        1.0, 1.0, 3.0, 3.0,
        5.0, 5.0, 7.0, 7.0,
        5.0, 5.0, 7.0, 7.0,
    ]]]).reshape(1, 4, 4)
    out = downsample_to_lr_grid(sr, scale_factor=4)
    assert out.shape == (1, 1, 1)
    assert out.item() == pytest.approx((1.0 + 3.0 + 5.0 + 7.0) / 4.0)


def test_multiband_downsamples_each_band_independently():
    sr = torch.stack([torch.full((4, 4), 2.0), torch.full((4, 4), 10.0)])
    out = downsample_to_lr_grid(sr, scale_factor=2)
    assert out.shape == (2, 2, 2)
    assert torch.allclose(out[0], torch.full((2, 2), 2.0))
    assert torch.allclose(out[1], torch.full((2, 2), 10.0))


def test_is_deterministic_across_repeated_calls():
    sr = torch.rand(4, 16, 16)
    a = downsample_to_lr_grid(sr, scale_factor=4)
    b = downsample_to_lr_grid(sr, scale_factor=4)
    assert torch.equal(a, b)


def test_scale_factor_of_one_is_identity():
    sr = torch.rand(3, 5, 5)
    out = downsample_to_lr_grid(sr, scale_factor=1)
    assert torch.allclose(out, sr)


def test_rejects_non_3d_input():
    sr = torch.rand(4, 8)  # missing the band axis
    with pytest.raises(ShapeMismatchError):
        downsample_to_lr_grid(sr, scale_factor=4)


def test_rejects_dimensions_not_divisible_by_scale_factor():
    sr = torch.rand(1, 9, 8)  # 9 is not divisible by 4
    with pytest.raises(ShapeMismatchError):
        downsample_to_lr_grid(sr, scale_factor=4)


def test_rejects_scale_factor_below_one():
    sr = torch.rand(1, 8, 8)
    with pytest.raises(ScaleFactorError):
        downsample_to_lr_grid(sr, scale_factor=0)


def test_rejects_non_integer_scale_factor():
    sr = torch.rand(1, 8, 8)
    with pytest.raises(ScaleFactorError):
        downsample_to_lr_grid(sr, scale_factor=2.5)


def test_method_name_constant_is_area_average_pool():
    assert AREA_AVERAGE_POOL == "area_average_pool"
