"""Tests for frame.analysis.indices -- NDVI computation (Phase 6).

NDVI = (NIR - Red) / (NIR + Red). Reuses frame.consistency.spectral_ratios'
already-tested, safe-division core (Phase 3) rather than reimplementing it;
this module adds the (bands, H, W) + band_names + validity-mask ergonomics
Phase 6 needs on top.
"""

import numpy as np
import torch
import pytest

from frame.analysis.errors import MissingBandError, ShapeMismatchError
from frame.analysis.indices import NDVIResult, compute_ndvi_from_stack

BANDS = ("B04", "B03", "B02", "B08")


def _stack(red, green, blue, nir, h=8, w=8):
    return torch.stack([torch.full((h, w), v) for v in (red, green, blue, nir)])


# ---------------------------------------------------------------------------
# 1. Known-value calculation
# ---------------------------------------------------------------------------

def test_ndvi_matches_known_formula():
    x = _stack(red=0.1, green=0.2, blue=0.05, nir=0.5)
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    expected = (0.5 - 0.1) / (0.5 + 0.1)
    assert torch.allclose(result.ndvi[result.valid_mask], torch.full_like(result.ndvi[result.valid_mask], expected), atol=1e-6)


def test_ndvi_negative_for_more_red_than_nir():
    x = _stack(red=0.4, green=0.2, blue=0.1, nir=0.1)
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    assert torch.all(result.ndvi[result.valid_mask] < 0)


def test_ndvi_positive_for_more_nir_than_red():
    x = _stack(red=0.1, green=0.2, blue=0.05, nir=0.6)
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    assert torch.all(result.ndvi[result.valid_mask] > 0)


def test_ndvi_range_is_bounded_between_minus_one_and_one():
    torch.manual_seed(0)
    x = torch.rand(4, 8, 8)
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    valid_values = result.ndvi[result.valid_mask]
    assert torch.all(valid_values >= -1.0 - 1e-6)
    assert torch.all(valid_values <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# 2. Zero-denominator handling
# ---------------------------------------------------------------------------

def test_zero_denominator_is_excluded_not_a_divide_by_zero_warning():
    x = _stack(red=0.0, green=0.1, blue=0.1, nir=0.0)  # red+nir == 0 everywhere
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning (e.g. RuntimeWarning) fails the test
        result = compute_ndvi_from_stack(x, band_names=BANDS)
    assert not torch.any(result.valid_mask)


def test_partial_zero_denominator_only_excludes_those_pixels():
    red = torch.zeros(4, 4)
    nir = torch.zeros(4, 4)
    nir[0, 0] = 0.3  # one computable pixel: (0.3-0)/(0.3+0) = 1.0
    x = torch.stack([red, torch.zeros(4, 4), torch.zeros(4, 4), nir])
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    assert result.valid_mask[0, 0].item() is True
    assert result.valid_mask.sum().item() == 1
    assert result.ndvi[0, 0].item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Masking (external validity mask, e.g. nodata/cloud)
# ---------------------------------------------------------------------------

def test_external_mask_is_combined_with_computability():
    x = _stack(red=0.1, green=0.2, blue=0.05, nir=0.5, h=4, w=4)  # every pixel computable
    external_mask = np.array([[True, False, True, True]] * 4)
    result = compute_ndvi_from_stack(x, band_names=BANDS, external_mask=external_mask)
    assert torch.equal(result.valid_mask, torch.from_numpy(external_mask))


# ---------------------------------------------------------------------------
# 5. Shape preservation
# ---------------------------------------------------------------------------

def test_ndvi_output_shape_matches_input_spatial_shape():
    x = _stack(0.1, 0.2, 0.05, 0.5, h=16, w=12)
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    assert result.ndvi.shape == (16, 12)
    assert result.valid_mask.shape == (16, 12)


# ---------------------------------------------------------------------------
# 7. Deterministic output
# ---------------------------------------------------------------------------

def test_deterministic_across_repeated_calls():
    torch.manual_seed(1)
    x = torch.rand(4, 8, 8)
    a = compute_ndvi_from_stack(x, band_names=BANDS)
    b = compute_ndvi_from_stack(x, band_names=BANDS)
    assert torch.equal(a.ndvi, b.ndvi)
    assert torch.equal(a.valid_mask, b.valid_mask)


# ---------------------------------------------------------------------------
# 10. No NaN/Inf leakage in valid output
# ---------------------------------------------------------------------------

def test_no_nan_or_inf_in_valid_pixels():
    torch.manual_seed(2)
    x = torch.rand(4, 16, 16)
    x[0, 0, 0] = 0.0
    x[3, 0, 0] = 0.0  # a genuinely non-computable pixel
    result = compute_ndvi_from_stack(x, band_names=BANDS)
    valid_values = result.ndvi[result.valid_mask]
    assert torch.isfinite(valid_values).all()


def test_metadata_records_resolution_and_grid_label():
    x = _stack(0.1, 0.2, 0.05, 0.5)
    result = compute_ndvi_from_stack(x, band_names=BANDS, resolution_m=10.0, grid_label="native_10m")
    assert isinstance(result, NDVIResult)
    assert result.resolution_m == 10.0
    assert result.grid_label == "native_10m"


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_missing_required_band_is_rejected():
    x = _stack(0.1, 0.2, 0.05, 0.5)
    with pytest.raises(MissingBandError):
        compute_ndvi_from_stack(x, band_names=("B02", "B03", "B04", "B05"))


def test_external_mask_shape_mismatch_is_rejected():
    x = _stack(0.1, 0.2, 0.05, 0.5, h=8, w=8)
    with pytest.raises(ShapeMismatchError):
        compute_ndvi_from_stack(x, band_names=BANDS, external_mask=np.ones((4, 4), dtype=bool))
