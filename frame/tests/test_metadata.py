"""Tests for frame.preprocessing.metadata.

RasterMetadata is the typed record carrying everything needed to describe
and later reconstruct the geospatial/provenance context of a preprocessed
input -- it must be constructible both when full geospatial info is known
(a real Sentinel-2 scene) and when it isn't (a synthetic unit-test array),
and it must never silently accept an internally-inconsistent record.
"""

import numpy as np
import pytest

from frame.preprocessing.errors import InvalidShapeError, MissingMetadataError
from frame.preprocessing.metadata import RasterMetadata, validate_metadata_matches_array


def test_unknown_factory_has_no_geospatial_fields():
    meta = RasterMetadata.unknown(
        band_names=("B04", "B03", "B02", "B08"),
        width=128,
        height=128,
        resolution_m=10.0,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    assert meta.crs is None
    assert meta.transform is None
    assert meta.bounds is None
    assert meta.acquisition_timestamp is None


def test_full_construction_preserves_all_fields():
    meta = RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0),
        bounds=(500000.0, 4398720.0, 501280.0, 4400000.0),
        resolution_m=10.0,
        width=128,
        height=128,
        band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=0.98,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    assert meta.crs == "EPSG:32630"
    assert meta.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0)
    assert meta.bounds == (500000.0, 4398720.0, 501280.0, 4400000.0)
    assert meta.band_names == ("B04", "B03", "B02", "B08")
    assert meta.cloud_mask_coverage == pytest.approx(0.98)


def test_metadata_is_immutable():
    meta = RasterMetadata.unknown(
        band_names=("B04",), width=1, height=1, resolution_m=10.0, sr_variant="x"
    )
    with pytest.raises(AttributeError):
        meta.crs = "EPSG:4326"  # type: ignore[misc]


def test_require_geospatial_passes_when_crs_and_transform_present():
    meta = RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        bounds=(0.0, 0.0, 1280.0, 1280.0),
        resolution_m=10.0,
        width=128,
        height=128,
        band_names=("B04", "B03", "B02", "B08"),
        acquisition_timestamp=None,
        nodata_value=None,
        cloud_mask_coverage=None,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )
    meta.require_geospatial()  # must not raise


def test_require_geospatial_raises_when_crs_missing():
    meta = RasterMetadata.unknown(
        band_names=("B04",), width=1, height=1, resolution_m=10.0, sr_variant="x"
    )
    with pytest.raises(MissingMetadataError):
        meta.require_geospatial()


def test_validate_metadata_matches_array_accepts_consistent_pair():
    meta = RasterMetadata.unknown(
        band_names=("B04", "B03", "B02", "B08"), width=128, height=128,
        resolution_m=10.0, sr_variant="x",
    )
    array = np.zeros((4, 128, 128), dtype="float32")
    validate_metadata_matches_array(meta, array)  # must not raise


def test_validate_metadata_matches_array_rejects_band_count_mismatch():
    meta = RasterMetadata.unknown(
        band_names=("B04", "B03"), width=128, height=128,  # says 2 bands
        resolution_m=10.0, sr_variant="x",
    )
    array = np.zeros((4, 128, 128), dtype="float32")  # actually has 4
    with pytest.raises(InvalidShapeError):
        validate_metadata_matches_array(meta, array)


def test_validate_metadata_matches_array_rejects_dimension_mismatch():
    meta = RasterMetadata.unknown(
        band_names=("B04",), width=64, height=64,  # says 64x64
        resolution_m=10.0, sr_variant="x",
    )
    array = np.zeros((1, 128, 128), dtype="float32")  # actually 128x128
    with pytest.raises(InvalidShapeError):
        validate_metadata_matches_array(meta, array)
