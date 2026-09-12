"""Tests for frame.preprocessing.validation.

These tests use small synthetic arrays only -- no network access, no
Hugging Face weights, no real Sentinel-2 data.
"""

import numpy as np
import pytest

from frame.preprocessing.errors import InvalidShapeError, UnsupportedBandsError, UnsupportedResolutionError
from frame.preprocessing.validation import (
    reorder_bands,
    validate_bands,
    validate_resolution,
    validate_shape,
)


# ---------------------------------------------------------------------------
# validate_shape
# ---------------------------------------------------------------------------

def test_validate_shape_accepts_correct_chw_array():
    array = np.zeros((4, 128, 128), dtype="float32")
    validate_shape(array, expected_size=128)  # must not raise


def test_validate_shape_rejects_wrong_ndim():
    array = np.zeros((128, 128), dtype="float32")  # missing band axis
    with pytest.raises(InvalidShapeError):
        validate_shape(array, expected_size=128)


def test_validate_shape_rejects_non_square_patch():
    array = np.zeros((4, 128, 64), dtype="float32")
    with pytest.raises(InvalidShapeError):
        validate_shape(array, expected_size=128)


def test_validate_shape_rejects_wrong_patch_size():
    array = np.zeros((4, 64, 64), dtype="float32")
    with pytest.raises(InvalidShapeError):
        validate_shape(array, expected_size=128)


def test_validate_shape_rejects_zero_sized_dimension():
    array = np.zeros((4, 0, 128), dtype="float32")
    with pytest.raises(InvalidShapeError):
        validate_shape(array, expected_size=None)


def test_validate_shape_allows_any_square_size_when_no_expected_size_given():
    array = np.zeros((4, 256, 256), dtype="float32")
    validate_shape(array, expected_size=None)  # must not raise


# ---------------------------------------------------------------------------
# validate_bands
# ---------------------------------------------------------------------------

def test_validate_bands_accepts_exact_match_any_order():
    validate_bands(["B08", "B02", "B04", "B03"], ["B04", "B03", "B02", "B08"])  # must not raise


def test_validate_bands_rejects_missing_band():
    with pytest.raises(UnsupportedBandsError):
        validate_bands(["B04", "B03", "B02"], ["B04", "B03", "B02", "B08"])


def test_validate_bands_rejects_missing_band_message_names_it():
    with pytest.raises(UnsupportedBandsError, match="B08"):
        validate_bands(["B04", "B03", "B02"], ["B04", "B03", "B02", "B08"])


def test_validate_bands_rejects_unexpected_extra_band():
    with pytest.raises(UnsupportedBandsError):
        validate_bands(["B04", "B03", "B02", "B08", "B11"], ["B04", "B03", "B02", "B08"])


def test_validate_bands_rejects_duplicate_band():
    with pytest.raises(UnsupportedBandsError):
        validate_bands(["B04", "B04", "B02", "B08"], ["B04", "B03", "B02", "B08"])


# ---------------------------------------------------------------------------
# reorder_bands
# ---------------------------------------------------------------------------

def test_reorder_bands_reindexes_to_expected_order():
    # band 0 filled with 0s, band 1 with 1s, band 2 with 2s, band 3 with 3s
    array = np.stack([np.full((2, 2), i, dtype="float32") for i in range(4)])
    band_names = ["B08", "B02", "B04", "B03"]  # NIR, Blue, Red, Green
    expected_order = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR

    reordered = reorder_bands(array, band_names, expected_order)

    assert reordered.shape == (4, 2, 2)
    # B04 was at index 2 in the input -> value 2
    assert np.all(reordered[0] == 2)
    # B03 was at index 3 in the input -> value 3
    assert np.all(reordered[1] == 3)
    # B02 was at index 1 in the input -> value 1
    assert np.all(reordered[2] == 1)
    # B08 was at index 0 in the input -> value 0
    assert np.all(reordered[3] == 0)


def test_reorder_bands_identity_when_already_in_order():
    array = np.stack([np.full((2, 2), i, dtype="float32") for i in range(4)])
    band_names = ["B04", "B03", "B02", "B08"]
    reordered = reorder_bands(array, band_names, band_names)
    assert np.array_equal(reordered, array)


def test_reorder_bands_raises_if_expected_band_absent():
    array = np.zeros((3, 2, 2), dtype="float32")
    with pytest.raises(UnsupportedBandsError):
        reorder_bands(array, ["B04", "B03", "B02"], ["B04", "B03", "B02", "B08"])


# ---------------------------------------------------------------------------
# validate_resolution
# ---------------------------------------------------------------------------

def test_validate_resolution_accepts_exact_match():
    validate_resolution(10.0, expected_resolution_m=10.0)  # must not raise


def test_validate_resolution_accepts_within_floating_point_tolerance():
    validate_resolution(10.0 + 1e-9, expected_resolution_m=10.0)  # must not raise


def test_validate_resolution_rejects_mismatch():
    with pytest.raises(UnsupportedResolutionError):
        validate_resolution(20.0, expected_resolution_m=10.0)


def test_validate_resolution_error_message_reports_both_values():
    with pytest.raises(UnsupportedResolutionError, match="20.*10|10.*20"):
        validate_resolution(20.0, expected_resolution_m=10.0)
