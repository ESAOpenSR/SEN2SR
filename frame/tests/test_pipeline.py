"""Tests for frame.preprocessing.pipeline.preprocess_rgbn -- the top-level
entry point that turns raw band arrays into a validated, model-ready
representation for the proven SEN2SRLite/NonReference_RGBN_x4 path.

All tests use small synthetic arrays. No network access, no model weights.
"""

import numpy as np
import pytest
import torch

from frame.preprocessing.errors import (
    InvalidShapeError,
    MissingMetadataError,
    UnsupportedBandsError,
    UnsupportedResolutionError,
)
from frame.preprocessing.masks import ValidityMask
from frame.preprocessing.pipeline import RGBN_BANDS, RGBN_RESOLUTION_M, preprocess_rgbn


def _digital_number_patch(bands=("B08", "B02", "B04", "B03"), size=128, fill=None):
    """A synthetic (bands, size, size) raw-digital-number array, out of order
    on purpose so reordering is actually exercised."""
    if fill is None:
        fill = {b: (i + 1) * 1000.0 for i, b in enumerate(bands)}
    array = np.stack([np.full((size, size), fill[b], dtype="float32") for b in bands])
    return array, list(bands)


def test_output_tensor_has_expected_shape_and_band_order():
    array, band_names = _digital_number_patch()
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0
    )
    assert isinstance(result.tensor, torch.Tensor)
    assert tuple(result.tensor.shape) == (4, 128, 128)
    assert result.metadata.band_names == RGBN_BANDS == ("B04", "B03", "B02", "B08")


def test_output_tensor_values_are_correctly_reordered_and_scaled():
    fill = {"B08": 4000.0, "B02": 2000.0, "B04": 1000.0, "B03": 3000.0}
    array, band_names = _digital_number_patch(bands=list(fill.keys()), fill=fill)
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0
    )
    # RGBN_BANDS order is (B04, B03, B02, B08) -> (1000, 3000, 2000, 4000)/10000
    expected = torch.tensor([0.1, 0.3, 0.2, 0.4]).view(4, 1, 1).expand(4, 128, 128)
    assert torch.allclose(result.tensor, expected, atol=1e-6)


def test_missing_required_band_is_rejected():
    array, band_names = _digital_number_patch(bands=("B08", "B02", "B04"))  # no B03
    with pytest.raises(UnsupportedBandsError):
        preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0)


def test_wrong_resolution_is_rejected():
    array, band_names = _digital_number_patch()
    with pytest.raises(UnsupportedResolutionError):
        preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=20.0)


def test_wrong_patch_size_is_rejected_by_default():
    array, band_names = _digital_number_patch(size=64)
    with pytest.raises(InvalidShapeError):
        preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0)


def test_patch_size_check_can_be_disabled_for_future_tiling_use():
    array, band_names = _digital_number_patch(size=64)
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number",
        resolution_m=10.0, patch_size=None,
    )
    assert tuple(result.tensor.shape) == (4, 64, 64)


def test_nan_and_inf_are_cleaned_before_reaching_the_model_input():
    array, band_names = _digital_number_patch()
    array[0, 0, 0] = np.nan
    array[1, 0, 0] = np.inf
    result = preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0)
    assert torch.isfinite(result.tensor).all()


def test_default_mask_is_all_valid_when_no_nodata_or_scl_given():
    array, band_names = _digital_number_patch()
    result = preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0)
    assert isinstance(result.mask, ValidityMask)
    assert result.mask.coverage() == pytest.approx(1.0)


def test_nodata_pixels_are_flagged_invalid_in_the_mask_not_in_the_tensor():
    array, band_names = _digital_number_patch()
    array[:, 0, 0] = 0.0  # nodata pixel across all bands
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number",
        resolution_m=10.0, nodata_value=0.0,
    )
    assert result.mask.array[0, 0] == False  # noqa: E712 (explicit bool check reads clearer here)
    assert result.mask.array[1, 1] == True  # noqa: E712
    assert result.metadata.cloud_mask_coverage == pytest.approx(result.mask.coverage())


def test_scl_cloud_pixels_are_flagged_invalid():
    array, band_names = _digital_number_patch()
    scl = np.full((128, 128), 4, dtype="uint8")  # vegetation everywhere
    scl[2, 2] = 9  # cloud high probability at one pixel
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number",
        resolution_m=10.0, scl=scl,
    )
    assert result.mask.array[2, 2] == False  # noqa: E712
    assert result.mask.array[0, 0] == True  # noqa: E712


def test_metadata_carries_sr_variant_and_dimensions():
    array, band_names = _digital_number_patch()
    result = preprocess_rgbn(array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0)
    assert result.metadata.sr_variant == "SEN2SRLite/NonReference_RGBN_x4"
    assert result.metadata.width == 128
    assert result.metadata.height == 128
    assert result.metadata.resolution_m == pytest.approx(RGBN_RESOLUTION_M)


def test_geospatial_metadata_is_preserved_when_provided():
    array, band_names = _digital_number_patch()
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0,
        crs="EPSG:32630", transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0),
        bounds=(500000.0, 4398720.0, 501280.0, 4400000.0),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
    )
    assert result.metadata.crs == "EPSG:32630"
    assert result.metadata.transform == (10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0)
    assert result.metadata.acquisition_timestamp == "2023-01-15T10:54:11.024000"


def test_require_geospatial_true_rejects_input_with_no_crs():
    array, band_names = _digital_number_patch()
    with pytest.raises(MissingMetadataError):
        preprocess_rgbn(
            array, band_names=band_names, input_scale="raw_digital_number",
            resolution_m=10.0, require_geospatial=True,
        )


def test_require_geospatial_true_accepts_input_with_crs_and_transform():
    array, band_names = _digital_number_patch()
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="raw_digital_number", resolution_m=10.0,
        crs="EPSG:32630", transform=(10.0, 0.0, 0.0, 0.0, -10.0, 0.0),
        bounds=(0.0, 0.0, 1280.0, 1280.0), require_geospatial=True,
    )
    assert result.metadata.crs == "EPSG:32630"


def test_already_reflectance_input_is_not_double_normalized():
    fill = {"B08": 0.4, "B02": 0.2, "B04": 0.1, "B03": 0.3}
    array, band_names = _digital_number_patch(bands=list(fill.keys()), fill=fill)
    result = preprocess_rgbn(
        array, band_names=band_names, input_scale="reflectance", resolution_m=10.0,
    )
    expected = torch.tensor([0.1, 0.3, 0.2, 0.4]).view(4, 1, 1).expand(4, 128, 128)
    assert torch.allclose(result.tensor, expected, atol=1e-6)
