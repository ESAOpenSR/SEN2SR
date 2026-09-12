"""Tests for frame.consistency.tiles -- cross-tile overlap consistency
(docs/FRAME_TECHNICAL_SPEC.md Section 9.2).

sen2sr.utils.predict_large does not expose pre-blend per-tile outputs (see
frame/consistency/tiles.py's module docstring and frame/consistency/README.md
for the verified limitation) -- this is tested here purely against synthetic
tile arrays, not a real predict_large run.
"""

import numpy as np
import pytest

from frame.consistency.errors import ShapeMismatchError
from frame.consistency.status import ComputationStatus
from frame.consistency.tiles import compare_tile_overlap


def _tile(value, bands=1, h=4, w=4):
    return np.full((bands, h, w), value, dtype="float32")


def test_identical_tiles_in_overlap_have_zero_discrepancy():
    tile_a = _tile(0.5, h=4, w=4)
    tile_b = _tile(0.5, h=4, w=4)
    # tile_b starts 2 pixels to the right -> a 4x2 overlap region
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 2))
    assert result.status == ComputationStatus.COMPUTABLE
    assert result.overlap_shape == (4, 2)
    assert result.mean_abs_discrepancy == pytest.approx(0.0)


def test_known_perturbation_in_overlap_returns_expected_discrepancy():
    tile_a = _tile(0.2, h=4, w=4)
    tile_b = _tile(0.5, h=4, w=4)
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 2))
    assert result.mean_abs_discrepancy == pytest.approx(0.3, abs=1e-6)
    assert result.rmse == pytest.approx(0.3, abs=1e-6)


def test_overlap_region_shape_is_correct_in_both_axes():
    tile_a = _tile(0.1, h=6, w=6)
    tile_b = _tile(0.1, h=6, w=6)
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(4, 4))
    assert result.overlap_shape == (2, 2)  # 6-4 in each axis


def test_non_overlapping_tiles_are_invalid_input():
    tile_a = _tile(0.1, h=4, w=4)
    tile_b = _tile(0.1, h=4, w=4)
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 10))
    assert result.status == ComputationStatus.INVALID_INPUT
    assert result.mean_abs_discrepancy is None


def test_band_count_mismatch_is_invalid_input():
    tile_a = _tile(0.1, bands=1)
    tile_b = _tile(0.1, bands=2)
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 2))
    assert result.status == ComputationStatus.INVALID_INPUT


def test_fully_masked_overlap_is_not_computable():
    tile_a = _tile(0.1, h=4, w=4)
    tile_b = _tile(0.9, h=4, w=4)
    mask_a = np.zeros((4, 4), dtype=bool)
    mask_b = np.ones((4, 4), dtype=bool)
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 2), mask_a=mask_a, mask_b=mask_b)
    assert result.status == ComputationStatus.NOT_COMPUTABLE
    assert result.valid_pixel_count == 0


def test_partially_masked_overlap_excludes_masked_pixels():
    tile_a = _tile(0.1, h=2, w=4)
    tile_b = _tile(0.1, h=2, w=4)
    tile_b[:, 0, 0] = 9.9  # would be a huge discrepancy at overlap-local (0,0) ...
    mask_b = np.ones((2, 4), dtype=bool)
    mask_b[0, 0] = False  # ... but that overlap-local pixel is masked out
    result = compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 0), mask_b=mask_b)
    assert result.status == ComputationStatus.COMPUTABLE
    assert result.valid_pixel_count == (2 * 4) - 1
    assert result.mean_abs_discrepancy == pytest.approx(0.0, abs=1e-6)


def test_rejects_non_3d_tile():
    tile_a = np.zeros((4, 4))  # missing band axis
    tile_b = _tile(0.1)
    with pytest.raises(ShapeMismatchError):
        compare_tile_overlap(tile_a, offset_a=(0, 0), tile_b=tile_b, offset_b=(0, 0))
