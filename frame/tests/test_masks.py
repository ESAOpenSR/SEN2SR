"""Tests for frame.preprocessing.masks.

A ValidityMask tracks, separately from the reflectance tensor, which pixels
are real observations vs. nodata/cloud/shadow -- so a masked pixel is never
confused with a genuine zero-reflectance pixel.
"""

import numpy as np
import pytest

from frame.preprocessing.masks import DEFAULT_SCL_INVALID_CLASSES, ValidityMask


def test_all_valid_mask_has_full_coverage():
    mask = ValidityMask.all_valid(shape=(3, 3))
    assert mask.array.shape == (3, 3)
    assert mask.array.dtype == bool
    assert np.all(mask.array)
    assert mask.coverage() == pytest.approx(1.0)


def test_from_nodata_flags_pixels_where_all_bands_equal_nodata():
    # 2 bands, 2x2 pixels. Pixel (0,0) is nodata in both bands -> invalid.
    # Pixel (0,1) is nodata in only one band -> still valid (a real
    # observation, since it isn't nodata across the whole band stack).
    array = np.array(
        [
            [[0.0, 0.0], [1.0, 2.0]],  # band 0
            [[0.0, 5.0], [1.0, 2.0]],  # band 1
        ],
        dtype="float32",
    )
    mask = ValidityMask.from_nodata(array, nodata_value=0.0)
    assert mask.array.tolist() == [[False, True], [True, True]]


def test_from_nodata_coverage_is_zero_when_everything_is_nodata():
    array = np.zeros((1, 2, 2), dtype="float32")  # everything nodata
    mask = ValidityMask.from_nodata(array, nodata_value=0.0)
    assert mask.coverage() == pytest.approx(0.0)  # coverage() == fraction VALID


def test_from_scl_flags_default_invalid_classes():
    # SCL codes: 4 = vegetation (valid), 9 = cloud high probability (invalid)
    scl = np.array([[4, 9], [4, 4]], dtype="uint8")
    mask = ValidityMask.from_scl(scl)
    assert mask.array.tolist() == [[True, False], [True, True]]


def test_from_scl_respects_custom_invalid_classes():
    scl = np.array([[4, 6]], dtype="uint8")  # 6 = water
    mask = ValidityMask.from_scl(scl, invalid_classes={6})
    assert mask.array.tolist() == [[True, False]]


def test_default_scl_invalid_classes_matches_esa_convention():
    # ESA Sentinel-2 L2A SCL codes: 0 no data, 1 saturated/defective,
    # 3 cloud shadow, 8 cloud medium prob, 9 cloud high prob, 10 thin cirrus.
    assert DEFAULT_SCL_INVALID_CLASSES == frozenset({0, 1, 3, 8, 9, 10})


def test_combine_is_invalid_wherever_any_input_mask_is_invalid():
    a = ValidityMask(np.array([[True, True], [False, True]]))
    b = ValidityMask(np.array([[True, False], [True, True]]))
    combined = a.combine(b)
    assert combined.array.tolist() == [[True, False], [False, True]]


def test_coverage_of_partially_valid_mask():
    mask = ValidityMask(np.array([[True, False], [True, True]]))
    assert mask.coverage() == pytest.approx(0.75)


def test_masked_pixel_is_distinguishable_from_real_zero_reflectance():
    # This is the core guarantee: a reflectance array that has been
    # zero-filled at a nodata pixel must remain flagged invalid in the
    # mask, never silently indistinguishable from a genuine 0.0 reading
    # at a *different*, valid pixel.
    reflectance = np.array([[[0.0, 0.0]]], dtype="float32")  # both pixels read 0.0
    mask = ValidityMask(np.array([[False, True]]))  # only the second is real
    assert reflectance[0, 0, 0] == reflectance[0, 0, 1] == 0.0
    assert mask.array[0, 0] != mask.array[0, 1]
