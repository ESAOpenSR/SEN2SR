"""Tests for frame.consistency.spectral_ratios -- inter-band relationship
(NDVI, B08/B04) comparison between the LR input and the downsampled SR
output (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).
"""

import numpy as np
import pytest

from frame.consistency.errors import MissingBandError
from frame.consistency.spectral_ratios import (
    compute_b08_b04_ratio_comparison,
    compute_ndvi,
    compute_ndvi_comparison,
    compute_simple_ratio,
)
from frame.consistency.status import ComputationStatus


# ---------------------------------------------------------------------------
# compute_ndvi / compute_simple_ratio (element-wise primitives)
# ---------------------------------------------------------------------------

def test_ndvi_matches_known_formula():
    nir = np.array([[0.5]])
    red = np.array([[0.1]])
    ndvi, computable = compute_ndvi(nir, red)
    assert computable[0, 0]
    assert ndvi[0, 0] == pytest.approx((0.5 - 0.1) / (0.5 + 0.1))


def test_ndvi_handles_divide_by_zero_safely():
    nir = np.array([[0.0]])
    red = np.array([[0.0]])
    ndvi, computable = compute_ndvi(nir, red)
    assert not computable[0, 0]
    # the array must not silently carry inf/garbage into downstream stats
    assert not np.isfinite(ndvi[0, 0]) or computable[0, 0] is False


def test_simple_ratio_matches_known_formula():
    ratio, computable = compute_simple_ratio(np.array([0.8]), np.array([0.2]))
    assert computable[0]
    assert ratio[0] == pytest.approx(4.0)


def test_simple_ratio_handles_divide_by_zero_safely():
    ratio, computable = compute_simple_ratio(np.array([0.8]), np.array([0.0]))
    assert not computable[0]


# ---------------------------------------------------------------------------
# compute_ndvi_comparison / compute_b08_b04_ratio_comparison
# ---------------------------------------------------------------------------

RGBN = ["B04", "B03", "B02", "B08"]


def _bands(red, green, blue, nir, size=2):
    return np.stack([np.full((size, size), v, dtype="float32") for v in (red, green, blue, nir)])


def test_ndvi_comparison_is_zero_when_lr_and_sr_agree():
    lr = _bands(0.1, 0.2, 0.05, 0.5)
    sr_down = _bands(0.1, 0.2, 0.05, 0.5)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.status == ComputationStatus.COMPUTABLE
    assert result.mean_abs_discrepancy == pytest.approx(0.0, abs=1e-6)


def test_ndvi_comparison_detects_a_known_discrepancy():
    lr = _bands(red=0.1, green=0.0, blue=0.0, nir=0.3)  # NDVI = (0.3-0.1)/(0.4) = 0.5
    sr_down = _bands(red=0.1, green=0.0, blue=0.0, nir=0.1)  # NDVI = 0 / 0.2 = 0.0
    mask = np.ones((2, 2), dtype=bool)
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.mean_abs_discrepancy == pytest.approx(0.5, abs=1e-6)


def test_b08_b04_ratio_comparison_is_zero_when_lr_and_sr_agree():
    lr = _bands(0.1, 0.2, 0.05, 0.4)
    sr_down = _bands(0.1, 0.2, 0.05, 0.4)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_b08_b04_ratio_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.status == ComputationStatus.COMPUTABLE
    assert result.mean_abs_discrepancy == pytest.approx(0.0, abs=1e-6)


def test_b08_b04_ratio_comparison_detects_a_known_discrepancy():
    lr = _bands(red=0.2, green=0.0, blue=0.0, nir=0.4)  # ratio = 2.0
    sr_down = _bands(red=0.2, green=0.0, blue=0.0, nir=0.8)  # ratio = 4.0
    mask = np.ones((2, 2), dtype=bool)
    result = compute_b08_b04_ratio_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.mean_abs_discrepancy == pytest.approx(2.0, abs=1e-6)


def test_masked_pixels_excluded_from_spectral_comparison():
    lr = _bands(red=0.1, green=0.0, blue=0.0, nir=0.3)
    sr_down = np.array(lr, copy=True)
    # corrupt one pixel's SR NIR value, but mask that pixel out
    sr_down[3, 0, 0] = 9.9
    mask = np.array([[False, True], [True, True]])
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.valid_pixel_count == 3
    assert result.mean_abs_discrepancy == pytest.approx(0.0, abs=1e-6)


def test_ndvi_comparison_not_computable_when_no_valid_pixels():
    lr = _bands(0.1, 0.0, 0.0, 0.3)
    sr_down = _bands(0.1, 0.0, 0.0, 0.3)
    mask = np.zeros((2, 2), dtype=bool)
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.status == ComputationStatus.NOT_COMPUTABLE
    assert result.valid_pixel_count == 0
    assert result.mean_abs_discrepancy is None


def test_ndvi_comparison_not_computable_when_denominator_always_zero():
    lr = _bands(0.0, 0.0, 0.0, 0.0)
    sr_down = _bands(0.0, 0.0, 0.0, 0.0)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.status == ComputationStatus.NOT_COMPUTABLE


def test_ndvi_comparison_reports_index_name():
    lr = _bands(0.1, 0.0, 0.0, 0.3)
    sr_down = _bands(0.1, 0.0, 0.0, 0.3)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_ndvi_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.index_name == "NDVI"


def test_b08_b04_ratio_comparison_reports_index_name():
    lr = _bands(0.1, 0.0, 0.0, 0.3)
    sr_down = _bands(0.1, 0.0, 0.0, 0.3)
    mask = np.ones((2, 2), dtype=bool)
    result = compute_b08_b04_ratio_comparison(lr, sr_down, mask, band_names=RGBN)
    assert result.index_name == "B08_B04_ratio"


def test_missing_required_band_is_rejected():
    lr = _bands(0.1, 0.0, 0.0, 0.3)
    sr_down = _bands(0.1, 0.0, 0.0, 0.3)
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(MissingBandError):
        compute_ndvi_comparison(lr, sr_down, mask, band_names=["B02", "B03", "B04", "B05"])
