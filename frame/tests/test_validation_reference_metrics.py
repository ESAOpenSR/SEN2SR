"""Tests for frame.validation.reference_metrics -- the standard reference-
based metric suite (PSNR, SSIM, RMSE, SAM, ERGAS) computed between an
estimate (bicubic OR SEN2SR output -- this module doesn't care which) and a
real HR reference image.

These are "(A) standard reference-based metrics" in the Phase 4 CRITICAL
three-way split -- distinct from opensr-test's own metrics (B) and from
frame.consistency's self-consistency diagnostics (C), see
frame/validation/report.py.
"""

import math

import numpy as np
import torch
import pytest

from frame.validation.errors import ShapeMismatchError
from frame.validation.reference_metrics import compute_reference_metrics
from frame.consistency.status import ComputationStatus

BANDS = ("B04", "B03", "B02", "B08")


def _stack(*values, h=16):
    return torch.stack([torch.full((h, h), v) for v in values])


def test_identical_images_have_near_zero_error_and_max_similarity():
    torch.manual_seed(0)
    hr = torch.rand(4, 16, 16) * 0.5 + 0.1  # avoid exact zeros (degenerate SAM norms)
    mask = np.ones((16, 16), dtype=bool)
    result = compute_reference_metrics(hr.clone(), hr, mask, band_names=BANDS, scale_factor=4)
    assert result.status == ComputationStatus.COMPUTABLE
    assert result.rmse == pytest.approx(0.0, abs=1e-5)
    assert result.sam_degrees == pytest.approx(0.0, abs=1e-2)
    assert result.ssim == pytest.approx(1.0, abs=1e-3)
    assert result.ergas == pytest.approx(0.0, abs=1e-3)
    assert result.psnr_db > 60  # near-exact match -> very high PSNR


def test_known_constant_offset_gives_exact_rmse_and_standard_psnr():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    est = _stack(0.2 + 0.05, 0.3 + 0.05, 0.1 + 0.05, 0.4 + 0.05)
    mask = np.ones((16, 16), dtype=bool)
    result = compute_reference_metrics(est, hr, mask, band_names=BANDS, scale_factor=4, data_range=1.0)
    assert result.rmse == pytest.approx(0.05, abs=1e-5)
    expected_psnr = 10 * math.log10(1.0**2 / (0.05**2))
    assert result.psnr_db == pytest.approx(expected_psnr, rel=1e-3)


def test_masked_pixels_excluded_from_every_metric():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    est = hr.clone()
    est[:, 0, 0] = 9.9  # huge outlier, but masked out below
    mask = np.ones((16, 16), dtype=bool)
    mask[0, 0] = False
    result = compute_reference_metrics(est, hr, mask, band_names=BANDS, scale_factor=4)
    assert result.rmse == pytest.approx(0.0, abs=1e-5)
    assert result.valid_pixel_count == 16 * 16 - 1


def test_all_invalid_mask_is_not_computable():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    est = hr.clone()
    mask = np.zeros((16, 16), dtype=bool)
    result = compute_reference_metrics(est, hr, mask, band_names=BANDS, scale_factor=4)
    assert result.status == ComputationStatus.NOT_COMPUTABLE
    assert result.valid_pixel_count == 0
    assert result.psnr_db is None
    assert result.ssim is None
    assert result.rmse is None
    assert result.sam_degrees is None
    assert result.ergas is None


def test_sam_matches_a_known_spectral_angle():
    # Two bands only, one pixel effectively (constant image): HR spectrum
    # (1, 0), estimate spectrum (0, 1) -> orthogonal vectors -> 90 degrees.
    hr = torch.stack([torch.full((8, 8), 1.0), torch.full((8, 8), 0.0)])
    est = torch.stack([torch.full((8, 8), 0.0), torch.full((8, 8), 1.0)])
    mask = np.ones((8, 8), dtype=bool)
    result = compute_reference_metrics(est, hr, mask, band_names=("B04", "B08"), scale_factor=4)
    assert result.sam_degrees == pytest.approx(90.0, abs=1e-3)


def test_ergas_is_zero_for_a_perfect_estimate():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    result = compute_reference_metrics(hr.clone(), hr, np.ones((16, 16), dtype=bool), band_names=BANDS, scale_factor=4)
    assert result.ergas == pytest.approx(0.0, abs=1e-3)


def test_ergas_matches_the_standard_formula_for_a_simple_case():
    # 1 band, HR mean = 0.2, RMSE = 0.02, scale_factor = 4
    # ERGAS = 100 * (1/scale) * sqrt(mean_over_bands((rmse/mean)^2))
    #       = 100 * 0.25 * (0.02/0.2) = 100 * 0.25 * 0.1 = 2.5
    hr = torch.full((1, 16, 16), 0.2)
    est = torch.full((1, 16, 16), 0.22)
    result = compute_reference_metrics(est, hr, np.ones((16, 16), dtype=bool), band_names=("B04",), scale_factor=4)
    assert result.ergas == pytest.approx(2.5, abs=1e-2)


def test_band_names_and_scale_factor_and_data_range_are_recorded():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    result = compute_reference_metrics(hr.clone(), hr, np.ones((16, 16), dtype=bool), band_names=BANDS, scale_factor=4)
    assert result.band_names == BANDS
    assert result.scale_factor == 4
    assert result.data_range == pytest.approx(1.0)


def test_shape_mismatch_between_estimate_and_hr_is_rejected():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    est = torch.rand(4, 8, 8)
    with pytest.raises(ShapeMismatchError):
        compute_reference_metrics(est, hr, np.ones((16, 16), dtype=bool), band_names=BANDS, scale_factor=4)


def test_mask_shape_mismatch_is_rejected():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    with pytest.raises(ShapeMismatchError):
        compute_reference_metrics(hr.clone(), hr, np.ones((8, 8), dtype=bool), band_names=BANDS, scale_factor=4)


def test_band_names_length_must_match_band_count():
    hr = _stack(0.2, 0.3, 0.1, 0.4)
    with pytest.raises(ShapeMismatchError):
        compute_reference_metrics(
            hr.clone(), hr, np.ones((16, 16), dtype=bool), band_names=("B04", "B08"), scale_factor=4
        )
