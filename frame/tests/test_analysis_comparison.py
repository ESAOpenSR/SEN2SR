"""Tests for frame.analysis.comparison -- comparing native-grid and SR-grid
NDVI on a common grid (Phase 6). Never compares a 2.5 m map directly against
a 10 m map -- the SR-grid NDVI is always downsampled first, reusing
frame.consistency's own area-average-pooling method (Phase 3), the
project's established "scientifically appropriate" grid-reduction approach.
"""

import numpy as np
import torch
import pytest

from frame.analysis.comparison import (
    NDVIComparisonResult,
    compare_ndvi_on_common_grid,
    downsample_ndvi_to_native_grid,
)
from frame.analysis.errors import ShapeMismatchError
from frame.consistency import ComputationStatus
from frame.consistency.downsample import AREA_AVERAGE_POOL


def test_downsample_reduces_by_the_scale_factor():
    sr_ndvi = torch.rand(16, 16)
    down = downsample_ndvi_to_native_grid(sr_ndvi, scale_factor=4)
    assert down.shape == (4, 4)


def test_downsample_of_constant_map_is_the_same_constant():
    sr_ndvi = torch.full((16, 16), 0.42)
    down = downsample_ndvi_to_native_grid(sr_ndvi, scale_factor=4)
    assert torch.allclose(down, torch.full((4, 4), 0.42), atol=1e-5)


def test_downsample_rejects_non_2d_input():
    with pytest.raises(ShapeMismatchError):
        downsample_ndvi_to_native_grid(torch.rand(1, 16, 16), scale_factor=4)


# ---------------------------------------------------------------------------
# compare_ndvi_on_common_grid -- comparison metrics
# ---------------------------------------------------------------------------

def test_perfect_agreement_gives_zero_error():
    native = torch.full((4, 4), 0.5)
    sr = torch.full((16, 16), 0.5)  # downsamples to the same constant
    mask = np.ones((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert isinstance(result.comparison, NDVIComparisonResult)
    assert result.comparison.status == ComputationStatus.COMPUTABLE
    assert result.comparison.rmse == pytest.approx(0.0, abs=1e-5)
    assert result.comparison.mean_abs_difference == pytest.approx(0.0, abs=1e-5)
    assert result.comparison.max_abs_difference == pytest.approx(0.0, abs=1e-5)


def test_known_constant_offset_gives_exact_metrics():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.4)  # downsamples to 0.4 -> offset 0.1 everywhere
    mask = np.ones((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.comparison.mean_abs_difference == pytest.approx(0.1, abs=1e-5)
    assert result.comparison.rmse == pytest.approx(0.1, abs=1e-5)
    assert result.comparison.max_abs_difference == pytest.approx(0.1, abs=1e-5)


def test_masked_pixels_are_excluded():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.3)
    sr[0:4, 12:16] = 0.9  # corrupts the last 4x4 SR block -> downsampled native[.,3] region
    mask = np.ones((4, 4), dtype=bool)
    mask[:, 3] = False  # exclude exactly that corrupted column
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.comparison.valid_pixel_count == 12
    assert result.comparison.mean_abs_difference == pytest.approx(0.0, abs=1e-5)


def test_valid_pixel_count_reported():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.3)
    mask = np.ones((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.comparison.valid_pixel_count == 16


def test_all_invalid_mask_is_not_computable():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.5)
    mask = np.zeros((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.comparison.status == ComputationStatus.NOT_COMPUTABLE
    assert result.comparison.mean_abs_difference is None
    assert result.comparison.rmse is None


def test_resampling_method_is_recorded_as_area_average_pool():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.3)
    mask = np.ones((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.comparison.resampling_method == AREA_AVERAGE_POOL


def test_difference_map_shape_matches_native_grid():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((16, 16), 0.4)
    mask = np.ones((4, 4), dtype=bool)
    result = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert result.absolute_difference_map.shape == (4, 4)
    assert result.sr_ndvi_downsampled.shape == (4, 4)


def test_shape_mismatch_between_native_grid_and_scale_factor_is_rejected():
    native = torch.full((4, 4), 0.3)
    sr = torch.full((15, 15), 0.3)  # not evenly divisible by 4, and not 4x native anyway
    mask = np.ones((4, 4), dtype=bool)
    with pytest.raises(ShapeMismatchError):
        compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)


def test_deterministic_across_repeated_calls():
    torch.manual_seed(0)
    native = torch.rand(4, 4)
    sr = torch.rand(16, 16)
    mask = np.ones((4, 4), dtype=bool)
    a = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    b = compare_ndvi_on_common_grid(native, sr, mask, scale_factor=4)
    assert torch.equal(a.absolute_difference_map, b.absolute_difference_map)
    assert a.comparison.rmse == b.comparison.rmse
