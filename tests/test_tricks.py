"""Tests for frequency filters and spectral hard constraints."""

import math

import pytest
import torch
from torch.nn import functional as F

from sen2sr.models.tricks import (
    FourierHardConstraint,
    HardConstraint,
    butterworth_filter,
    gaussian_filter,
    ideal_filter,
    sigmoid_filter,
)


def test_ideal_filter_retains_frequencies_inside_cutoff():
    expected = torch.zeros(5, 7)
    expected[2, 2:5] = 1
    expected[1:4, 3] = 1

    torch.testing.assert_close(ideal_filter((5, 7), cutoff=1), expected)


def test_butterworth_filter_has_expected_response_at_cutoff():
    result = butterworth_filter((9, 9), cutoff=2, order=2)

    assert result[4, 4] == 1
    assert result[4, 6] == 0.5
    torch.testing.assert_close(result[4, 8], torch.tensor(1 / 17))


def test_gaussian_filter_has_unit_center_and_expected_falloff():
    result = gaussian_filter((5, 7), cutoff=2)

    assert result.shape == (5, 7)
    assert result[2, 3] == 1
    torch.testing.assert_close(result[2, 5], torch.tensor(math.exp(-0.5)))


def test_sigmoid_filter_has_half_response_at_cutoff():
    result = sigmoid_filter((5, 7), cutoff=2, sharpness=1)

    assert result.shape == (5, 7)
    assert result[2, 5] == 0.5
    assert result[2, 3] > result[2, 4] > result[2, 5] > result[2, 6]


@pytest.mark.parametrize("mask_value", [0, 1])
def test_hard_constraint_selects_sr_or_upsampled_reference(mask_value):
    low_resolution = torch.arange(32, dtype=torch.float32).reshape(1, 2, 4, 4)
    prediction = torch.arange(128, dtype=torch.float32).reshape(1, 2, 8, 8) / 3
    model = HardConstraint(torch.full((8, 8), mask_value, dtype=torch.float32))

    result = model(low_resolution, prediction)

    expected = F.interpolate(low_resolution, size=(8, 8), mode="bicubic", antialias=True) if mask_value else prediction
    torch.testing.assert_close(result, expected)


def test_hard_constraint_can_select_reference_bands():
    low_resolution = torch.tensor([2.0, 5.0, 9.0]).reshape(1, 3, 1, 1).expand(1, 3, 4, 4)
    model = HardConstraint(torch.ones(8, 8), bands=[2, 0])

    result = model(low_resolution, torch.zeros(1, 2, 8, 8))

    expected = torch.tensor([9.0, 2.0]).reshape(1, 2, 1, 1).expand_as(result)
    torch.testing.assert_close(result, expected)


def test_fourier_constraint_preserves_sr_detail_and_reference_mean():
    rows, columns = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    detail = ((rows + columns) % 2 * 2 - 1).float()[None, None]
    model = FourierHardConstraint("ideal", {}, sr_image_size=(8, 8), scale_factor=2)

    result = model(torch.full((1, 1, 4, 4), 3.0), detail)

    torch.testing.assert_close(result, detail + 3)


def test_fourier_constraint_rejects_unknown_filter():
    with pytest.raises(ValueError, match="Unsupported fourier_method"):
        FourierHardConstraint("unknown", {}, sr_image_size=(8, 8), scale_factor=2)


@pytest.mark.parametrize(
    "method,parameters",
    [("ideal", {}), ("butterworth", {"order": 2}), ("gaussian", {}), ("sigmoid", {"sharpness": 1})],
)
def test_fourier_constraint_supports_each_filter(method, parameters):
    model = FourierHardConstraint(method, parameters, sr_image_size=(8, 8), scale_factor=2)

    result = model(torch.ones(1, 2, 4, 4), torch.zeros(1, 2, 8, 8))

    assert result.shape == (1, 2, 8, 8)
    assert torch.isfinite(result).all()
