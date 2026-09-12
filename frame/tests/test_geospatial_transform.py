"""Tests for frame.geospatial.transform -- deterministic output-transform
derivation for the SR pipeline (docs/FRAME_TECHNICAL_SPEC.md Section 10).

Core invariants under test:
  - same CRS as input
  - same geographic footprint/bounds
  - pixel size divided by the SR scale factor
  - origin preserved
  - rotation/shear terms scaled consistently (not dropped)
"""

import math

import pytest

from frame.geospatial.errors import MissingCRSError, MissingTransformError
from frame.geospatial.transform import RGBN_SCALE_FACTOR, derive_output_metadata, derive_output_transform
from frame.preprocessing.metadata import RasterMetadata


def _isclose(a, b, tol=1e-6):
    return math.isclose(a, b, abs_tol=tol)


def _isclose_seq(a, b, tol=1e-6):
    return all(_isclose(x, y, tol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# derive_output_transform
# ---------------------------------------------------------------------------

def test_pixel_size_divided_by_scale_factor():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0)
    out = derive_output_transform(transform, scale_factor=4)
    assert _isclose(out[0], 2.5)  # pixel width
    assert _isclose(out[4], -2.5)  # pixel height (still negative -> north-up)


def test_origin_is_preserved():
    transform = (10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0)
    out = derive_output_transform(transform, scale_factor=4)
    assert _isclose(out[2], 720285.0)  # c: x-origin unchanged
    assert _isclose(out[5], 4375125.0)  # f: y-origin unchanged


def test_rotation_and_shear_terms_are_scaled_not_dropped():
    # b and d are non-zero (a rotated/sheared grid) -- they must scale down
    # by the same factor as a/e, not be zeroed out.
    transform = (10.0, 2.0, 0.0, 3.0, -10.0, 0.0)
    out = derive_output_transform(transform, scale_factor=4)
    assert _isclose(out[1], 0.5)  # b / 4
    assert _isclose(out[3], 0.75)  # d / 4


def test_non_square_pixels_scale_independently_per_axis():
    transform = (10.0, 0.0, 0.0, 0.0, -20.0, 0.0)  # 10m x 20m pixels
    out = derive_output_transform(transform, scale_factor=4)
    assert _isclose(out[0], 2.5)
    assert _isclose(out[4], -5.0)


def test_scale_factor_of_one_is_identity():
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0)
    out = derive_output_transform(transform, scale_factor=1)
    assert _isclose_seq(out, transform)


# ---------------------------------------------------------------------------
# derive_output_metadata
# ---------------------------------------------------------------------------

def _input_metadata(**overrides):
    defaults = dict(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4373845.0, 721565.0, 4375125.0),
        resolution_m=10.0,
        width=128,
        height=128,
        band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    defaults.update(overrides)
    return RasterMetadata(**defaults)


def test_output_metadata_preserves_crs():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=RGBN_SCALE_FACTOR, output_band_names=meta.band_names)
    assert out.crs == meta.crs


def test_output_metadata_footprint_matches_input_footprint():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=RGBN_SCALE_FACTOR, output_band_names=meta.band_names)
    assert _isclose_seq(out.bounds, meta.bounds)


def test_output_metadata_width_height_scale_up_by_factor():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=4, output_band_names=meta.band_names)
    assert out.width == meta.width * 4
    assert out.height == meta.height * 4


def test_output_pixel_size_is_input_pixel_size_over_scale_factor():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=4, output_band_names=meta.band_names)
    assert _isclose(out.resolution_m, meta.resolution_m / 4)


def test_output_metadata_preserves_provenance_fields():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=4, output_band_names=meta.band_names)
    assert out.acquisition_timestamp == meta.acquisition_timestamp
    assert out.sr_variant == meta.sr_variant
    assert out.nodata_value == meta.nodata_value
    assert out.cloud_mask_coverage == meta.cloud_mask_coverage


def test_output_metadata_uses_given_output_band_names():
    meta = _input_metadata()
    out = derive_output_metadata(meta, scale_factor=4, output_band_names=("R", "G", "B", "N"))
    assert out.band_names == ("R", "G", "B", "N")


def test_output_metadata_rejects_missing_crs():
    meta = _input_metadata(crs=None)
    with pytest.raises(MissingCRSError):
        derive_output_metadata(meta, scale_factor=4, output_band_names=meta.band_names)


def test_output_metadata_rejects_missing_transform():
    meta = _input_metadata(transform=None)
    with pytest.raises(MissingTransformError):
        derive_output_metadata(meta, scale_factor=4, output_band_names=meta.band_names)


def test_rgbn_scale_factor_constant_is_four():
    assert RGBN_SCALE_FACTOR == 4
