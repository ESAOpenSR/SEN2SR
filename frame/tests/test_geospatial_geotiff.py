"""Tests for frame.geospatial.geotiff -- write/read GeoTIFF round-trip.

Uses small synthetic rasters and a pytest tmp_path -- no network, no model
weights.
"""

import math

import numpy as np
import pytest

from frame.geospatial.errors import MissingCRSError, MissingTransformError
from frame.geospatial.geotiff import read_geotiff, write_geotiff
from frame.preprocessing.metadata import RasterMetadata


def _isclose_seq(a, b, tol=1e-6):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))


def _metadata(**overrides):
    defaults = dict(
        crs="EPSG:32630",
        transform=(2.5, 0.0, 720285.0, 0.0, -2.5, 4375125.0),
        bounds=(720285.0, 4374805.0, 720605.0, 4375125.0),
        resolution_m=2.5,
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


def _synthetic_array(bands=4, size=128):
    rng = np.random.default_rng(0)
    return rng.random((bands, size, size), dtype="float32")


# ---------------------------------------------------------------------------
# write_geotiff rejects missing geospatial context
# ---------------------------------------------------------------------------

def test_write_rejects_missing_crs(tmp_path):
    array = _synthetic_array()
    meta = _metadata(crs=None)
    with pytest.raises(MissingCRSError):
        write_geotiff(tmp_path / "out.tif", array, meta)


def test_write_rejects_missing_transform(tmp_path):
    array = _synthetic_array()
    meta = _metadata(transform=None)
    with pytest.raises(MissingTransformError):
        write_geotiff(tmp_path / "out.tif", array, meta)


# ---------------------------------------------------------------------------
# round-trip write -> read
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_crs(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert read_meta.crs == "EPSG:32630"


def test_roundtrip_preserves_transform(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert _isclose_seq(read_meta.transform, meta.transform)


def test_roundtrip_preserves_bounds(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert _isclose_seq(read_meta.bounds, meta.bounds)


def test_roundtrip_preserves_dimensions(tmp_path):
    array = _synthetic_array(size=64)
    meta = _metadata(width=64, height=64)
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert read_meta.width == 64
    assert read_meta.height == 64


def test_roundtrip_preserves_pixel_values(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    read_array, _ = read_geotiff(path)
    assert read_array.shape == array.shape
    assert np.allclose(read_array, array, atol=1e-6)


def test_roundtrip_preserves_band_order_via_descriptions(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert read_meta.band_names == ("B04", "B03", "B02", "B08")


def test_roundtrip_preserves_nodata(tmp_path):
    array = _synthetic_array()
    meta = _metadata(nodata_value=0.0)
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert read_meta.nodata_value == pytest.approx(0.0)


def test_roundtrip_preserves_model_and_acquisition_tags(tmp_path):
    array = _synthetic_array()
    meta = _metadata()
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert read_meta.sr_variant == "SEN2SRLite/NonReference_RGBN_x4"
    assert read_meta.acquisition_timestamp == "2023-01-15T10:54:11.024000"


def test_roundtrip_with_nonzero_origin(tmp_path):
    array = _synthetic_array()
    meta = _metadata(transform=(2.5, 0.0, 999999.0, 0.0, -2.5, -888888.0))
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert math.isclose(read_meta.transform[2], 999999.0, abs_tol=1e-6)
    assert math.isclose(read_meta.transform[5], -888888.0, abs_tol=1e-6)


def test_roundtrip_with_non_square_pixels(tmp_path):
    array = _synthetic_array(size=32)
    meta = _metadata(
        transform=(2.5, 0.0, 0.0, 0.0, -5.0, 0.0), width=32, height=32,
        bounds=(0.0, -160.0, 80.0, 0.0),
    )
    path = tmp_path / "out.tif"
    write_geotiff(path, array, meta)
    _, read_meta = read_geotiff(path)
    assert math.isclose(read_meta.transform[0], 2.5, abs_tol=1e-6)
    assert math.isclose(read_meta.transform[4], -5.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# read_geotiff on a file with no CRS
# ---------------------------------------------------------------------------

def test_read_requires_crs_by_default(tmp_path):
    import rasterio

    path = tmp_path / "no_crs.tif"
    array = _synthetic_array(bands=1, size=4)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32"
    ) as dst:
        dst.write(array)

    with pytest.raises(MissingCRSError):
        read_geotiff(path)


def test_read_allows_missing_crs_when_not_required(tmp_path):
    import rasterio

    path = tmp_path / "no_crs.tif"
    array = _synthetic_array(bands=1, size=4)
    with rasterio.open(
        path, "w", driver="GTiff", height=4, width=4, count=1, dtype="float32"
    ) as dst:
        dst.write(array)

    _, read_meta = read_geotiff(path, require_crs=False)
    assert read_meta.crs is None
