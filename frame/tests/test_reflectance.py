"""Tests for frame.preprocessing.reflectance.

Reproduces the proven Baseline 0 / upstream README normalization behavior:
    (digital_number / 10000).astype(float32), then NaN/Inf -> 0.0
and adds an explicit, non-guessing path for already-normalized input.
"""

import numpy as np
import pytest

from frame.preprocessing.errors import InvalidInputScaleError
from frame.preprocessing.reflectance import to_reflectance


def test_raw_digital_number_is_scaled_by_10000():
    array = np.array([[[10000.0, 5000.0]]], dtype="float64")  # shape (1,1,2)
    result = to_reflectance(array, input_scale="raw_digital_number")
    assert np.allclose(result, [[[1.0, 0.5]]])


def test_output_is_always_float32():
    array = np.array([[[10000.0]]], dtype="float64")
    result = to_reflectance(array, input_scale="raw_digital_number")
    assert result.dtype == np.float32


def test_already_reflectance_scale_is_not_divided_again():
    # Values already in [0, 1] reflectance space must pass through unchanged
    # in magnitude -- dividing by 10000 again would be silent double
    # normalization.
    array = np.array([[[0.42, 0.13]]], dtype="float32")
    result = to_reflectance(array, input_scale="reflectance")
    assert np.allclose(result, [[[0.42, 0.13]]])


def test_nan_is_replaced_with_zero():
    array = np.array([[[np.nan, 5000.0]]], dtype="float64")
    result = to_reflectance(array, input_scale="raw_digital_number")
    assert np.allclose(result, [[[0.0, 0.5]]])


def test_positive_infinity_is_replaced_with_zero():
    array = np.array([[[np.inf, 5000.0]]], dtype="float64")
    result = to_reflectance(array, input_scale="raw_digital_number")
    assert np.allclose(result, [[[0.0, 0.5]]])


def test_negative_infinity_is_replaced_with_zero():
    array = np.array([[[-np.inf, 5000.0]]], dtype="float64")
    result = to_reflectance(array, input_scale="raw_digital_number")
    assert np.allclose(result, [[[0.0, 0.5]]])


def test_nan_cleanup_also_applies_to_already_reflectance_input():
    array = np.array([[[np.nan, 0.3]]], dtype="float32")
    result = to_reflectance(array, input_scale="reflectance")
    assert np.allclose(result, [[[0.0, 0.3]]])


def test_unknown_input_scale_raises_clear_error():
    array = np.zeros((1, 1, 1), dtype="float32")
    with pytest.raises(InvalidInputScaleError, match="banana"):
        to_reflectance(array, input_scale="banana")
