"""Tests for frame.geospatial.metadata.

frame.geospatial reuses frame.preprocessing.metadata.RasterMetadata as its
raster/metadata representation (it already carries CRS, affine transform,
bounds, resolution, width, height, band names, and nodata -- see
frame/geospatial/README.md for why this is not duplicated) and adds:
  - bounds_from_transform: general (rotation-aware) footprint computation
  - validate_geospatial_completeness: the "no silently fabricated CRS" gate
"""

import math

import pytest

from frame.geospatial.errors import MissingCRSError, MissingTransformError
from frame.geospatial.metadata import bounds_from_transform, validate_geospatial_completeness
from frame.preprocessing.metadata import RasterMetadata


def _isclose_bounds(a, b, tol=1e-6):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# bounds_from_transform
# ---------------------------------------------------------------------------

def test_bounds_from_transform_simple_north_up():
    # 10 m pixels, origin at (500000, 4400000), no rotation, 128x128 pixels
    transform = (10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0)
    bounds = bounds_from_transform(transform, width=128, height=128)
    assert _isclose_bounds(bounds, (500000.0, 4398720.0, 501280.0, 4400000.0))


def test_bounds_from_transform_with_nonzero_origin():
    transform = (10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0)
    bounds = bounds_from_transform(transform, width=128, height=128)
    assert _isclose_bounds(bounds, (720285.0, 4373845.0, 721565.0, 4375125.0))


def test_bounds_from_transform_non_square_pixels():
    # 10 m pixel width, 20 m pixel height
    transform = (10.0, 0.0, 0.0, 0.0, -20.0, 1000.0)
    bounds = bounds_from_transform(transform, width=100, height=50)
    assert _isclose_bounds(bounds, (0.0, 0.0, 1000.0, 1000.0))


def test_bounds_from_transform_with_rotation():
    # A 45-degree-rotated pixel grid: a=b=7.0710678 (~10/sqrt(2)) style shear.
    # Corners must be mapped individually, not assumed axis-aligned.
    transform = (5.0, 5.0, 0.0, 5.0, -5.0, 0.0)
    bounds = bounds_from_transform(transform, width=10, height=10)
    # corners: (0,0)->(0,0); (10,0)->(50,50); (0,10)->(50,-50); (10,10)->(100,0)
    assert _isclose_bounds(bounds, (0.0, -50.0, 100.0, 50.0))


# ---------------------------------------------------------------------------
# validate_geospatial_completeness
# ---------------------------------------------------------------------------

def _metadata(crs=None, transform=None):
    return RasterMetadata(
        crs=crs,
        transform=transform,
        bounds=None,
        resolution_m=10.0,
        width=1,
        height=1,
        band_names=("B04",),
        acquisition_timestamp=None,
        nodata_value=None,
        cloud_mask_coverage=None,
        sr_variant="x",
    )


def test_validate_geospatial_completeness_accepts_full_metadata():
    meta = _metadata(crs="EPSG:32630", transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
    validate_geospatial_completeness(meta)  # must not raise


def test_validate_geospatial_completeness_rejects_missing_crs():
    meta = _metadata(crs=None, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
    with pytest.raises(MissingCRSError):
        validate_geospatial_completeness(meta)


def test_validate_geospatial_completeness_rejects_missing_transform():
    meta = _metadata(crs="EPSG:32630", transform=None)
    with pytest.raises(MissingTransformError):
        validate_geospatial_completeness(meta)


def test_validate_geospatial_completeness_never_fabricates_a_default_crs():
    # Regression guard: the function must raise, not return/patch in a
    # default CRS such as EPSG:4326.
    meta = _metadata(crs=None, transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0))
    with pytest.raises(MissingCRSError, match="CRS"):
        validate_geospatial_completeness(meta)
