"""Tests for frame.validation.bicubic -- the naive baseline every SR result
is compared against (docs/FRAME_TECHNICAL_SPEC.md Section 11's "bicubic
baseline" convention, matching what experiments/baseline/run_baseline.py and
experiments/consistency/run_experiment.py already use for visualization).
"""

import torch
import pytest

from frame.validation.bicubic import bicubic_upsample
from frame.validation.errors import ShapeMismatchError


def test_output_shape_matches_scale_factor():
    lr = torch.rand(4, 8, 8)
    out = bicubic_upsample(lr, scale_factor=4)
    assert out.shape == (4, 32, 32)


def test_constant_image_upsamples_to_the_same_constant():
    lr = torch.full((1, 8, 8), 0.4)
    out = bicubic_upsample(lr, scale_factor=4)
    assert torch.allclose(out, torch.full((1, 32, 32), 0.4), atol=1e-5)


def test_multiband_upsamples_each_band_independently():
    lr = torch.stack([torch.full((8, 8), 0.1), torch.full((8, 8), 0.9)])
    out = bicubic_upsample(lr, scale_factor=2)
    assert out.shape == (2, 16, 16)
    assert torch.allclose(out[0], torch.full((16, 16), 0.1), atol=1e-5)
    assert torch.allclose(out[1], torch.full((16, 16), 0.9), atol=1e-5)


def test_is_deterministic():
    lr = torch.rand(4, 8, 8)
    a = bicubic_upsample(lr, scale_factor=4)
    b = bicubic_upsample(lr, scale_factor=4)
    assert torch.equal(a, b)


def test_rejects_non_3d_input():
    lr = torch.rand(8, 8)
    with pytest.raises(ShapeMismatchError):
        bicubic_upsample(lr, scale_factor=4)


def test_rejects_scale_factor_below_one():
    lr = torch.rand(4, 8, 8)
    with pytest.raises(ShapeMismatchError):
        bicubic_upsample(lr, scale_factor=0)
