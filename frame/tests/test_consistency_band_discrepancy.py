"""Tests for frame.consistency.band_discrepancy -- the downsample-consistency
diagnostic (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).

This is a SELF-consistency test: it compares an SR output, reduced back to
its LR grid, against the real LR observation it was derived from. It is NOT
an accuracy metric against ground truth (there is no ground truth here).
"""

import numpy as np
import torch
import pytest

from frame.consistency.band_discrepancy import compute_downsample_consistency
from frame.consistency.errors import InvalidMaskError, ScaleFactorError, ShapeMismatchError
from frame.consistency.status import ComputationStatus


def _lr_sr_pair(lr_value: float, sr_value: float, size: int = 2, scale: int = 4, bands: int = 1):
    lr = torch.full((bands, size, size), lr_value)
    sr = torch.full((bands, size * scale, size * scale), sr_value)
    return lr, sr


def test_perfect_consistency_returns_zero_error():
    lr, sr = _lr_sr_pair(0.3, 0.3)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.status == ComputationStatus.COMPUTABLE
    assert result.overall.mean_abs_error == pytest.approx(0.0, abs=1e-6)
    assert result.overall.rmse == pytest.approx(0.0, abs=1e-6)
    assert result.overall.max_abs_error == pytest.approx(0.0, abs=1e-6)


def test_known_perturbation_returns_expected_error():
    # LR is a constant 0.2; SR downsamples to a constant 0.5 everywhere ->
    # every valid pixel has |error| = 0.3 exactly.
    lr, sr = _lr_sr_pair(0.2, 0.5)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.mean_abs_error == pytest.approx(0.3, abs=1e-6)
    assert result.overall.rmse == pytest.approx(0.3, abs=1e-6)
    assert result.overall.max_abs_error == pytest.approx(0.3, abs=1e-6)


def test_per_band_results_are_correct_and_independent():
    lr = torch.stack([torch.full((2, 2), 0.1), torch.full((2, 2), 0.9)])
    sr = torch.stack([torch.full((8, 8), 0.1), torch.full((8, 8), 0.4)])  # band 0 perfect, band 1 off by 0.5
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04", "B08"], scale_factor=4)
    assert result.per_band["B04"].mean_abs_error == pytest.approx(0.0, abs=1e-6)
    assert result.per_band["B08"].mean_abs_error == pytest.approx(0.5, abs=1e-6)


def test_masked_pixels_are_excluded_from_statistics():
    # One pixel has a huge discrepancy but is masked out; the other is
    # perfect. Only the valid pixel should count.
    lr = torch.tensor([[[0.1, 0.1], [0.1, 0.1]]])
    sr_block = torch.zeros(1, 8, 8)
    sr_block[:, 0:4, 0:4] = 0.1  # top-left LR pixel: perfect match
    sr_block[:, 0:4, 4:8] = 9.9  # top-right LR pixel: huge discrepancy, but masked out below
    sr_block[:, 4:8, 0:4] = 0.1
    sr_block[:, 4:8, 4:8] = 0.1
    mask = np.array([[True, False], [True, True]])
    result = compute_downsample_consistency(lr, sr_block, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.valid_pixel_count == 3
    assert result.overall.mean_abs_error == pytest.approx(0.0, abs=1e-6)


def test_all_invalid_input_is_not_computable():
    lr, sr = _lr_sr_pair(0.2, 0.9)
    mask = np.zeros((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.status == ComputationStatus.NOT_COMPUTABLE
    assert result.overall.valid_pixel_count == 0
    assert result.overall.mean_abs_error is None
    assert result.overall.rmse is None
    assert result.per_band["B04"].status == ComputationStatus.NOT_COMPUTABLE
    assert result.per_band["B04"].mean_abs_error is None


def test_normalized_rmse_is_none_when_lr_mean_is_zero():
    lr, sr = _lr_sr_pair(0.0, 0.3)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.normalized_rmse is None


def test_normalized_rmse_is_computed_when_lr_mean_is_nonzero():
    lr, sr = _lr_sr_pair(0.2, 0.3)  # rmse=0.1, lr mean abs = 0.2 -> normalized = 0.5
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.overall.normalized_rmse == pytest.approx(0.5, abs=1e-6)


def test_valid_pixel_count_and_mask_coverage_reported_at_top_level():
    lr, sr = _lr_sr_pair(0.2, 0.2, size=2)
    mask = np.array([[True, True], [True, False]])
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.valid_pixel_count == 3
    assert result.total_pixel_count == 4
    assert result.mask_coverage == pytest.approx(0.75)


def test_reproducibility_metadata_is_present():
    lr, sr = _lr_sr_pair(0.2, 0.2)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
    assert result.downsample_method == "area_average_pool"
    assert result.scale_factor == 4
    assert result.band_names == ("B04",)


def test_shape_mismatch_between_lr_and_sr_is_rejected():
    lr = torch.zeros(1, 2, 2)
    sr = torch.zeros(1, 9, 9)  # not 2*scale_factor in either dimension for any integer scale_factor consistent with lr
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises((ShapeMismatchError, ScaleFactorError)):
        compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)


def test_sr_scale_factor_must_exactly_match_lr_times_scale_factor():
    # SR is evenly divisible by scale_factor (8/4=2) but that does NOT equal
    # the LR grid (2x2) -- must be caught as a scale-factor/shape mismatch,
    # not silently downsampled to the wrong grid size.
    lr = torch.zeros(1, 2, 2)
    sr = torch.zeros(1, 8, 8)
    wrong_lr = torch.zeros(1, 3, 3)
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ShapeMismatchError):
        compute_downsample_consistency(wrong_lr, sr, mask, band_names=["B04"], scale_factor=4)


def test_band_count_mismatch_is_rejected():
    lr = torch.zeros(1, 2, 2)
    sr = torch.zeros(2, 8, 8)  # 2 bands vs LR's 1 band
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ShapeMismatchError):
        compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)


def test_band_names_length_must_match_band_count():
    lr, sr = _lr_sr_pair(0.2, 0.2)
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(ShapeMismatchError):
        compute_downsample_consistency(lr, sr, mask, band_names=["B04", "B08"], scale_factor=4)


def test_mask_shape_mismatch_is_rejected():
    lr, sr = _lr_sr_pair(0.2, 0.2)
    mask = np.ones((3, 3), dtype=bool)  # doesn't match the 2x2 LR grid
    with pytest.raises(InvalidMaskError):
        compute_downsample_consistency(lr, sr, mask, band_names=["B04"], scale_factor=4)
