"""Tests for frame.validation.opensr_test_adapter.

Uses a synthetic dict shaped exactly like a real, loaded opensr-test subset
(verified against the real `spot` subset during development: keys
L2A/L1C/HR/HRharm/metadata; L2A shape (N,12,H,W); L1C shape (N,13,H,W); HR/
HRharm shape (N,4,H*scale,W*scale); metadata a pandas DataFrame with columns
lr_file/hr_file/roi/lr_gee_id/reflectance/spectral/spatial/crs/affine). No
network access, no real download -- see
frame/validation/README.md for how these facts were verified against the
real dataset.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from frame.validation.errors import (
    BenchmarkFormatError,
    SampleIndexError,
    UnsupportedSubsetError,
)
from frame.validation.opensr_test_adapter import (
    L2A_BAND_ORDER,
    RGBN_BAND_NAMES,
    SUPPORTED_SUBSETS,
    extract_sample,
)

N, H, SCALE = 3, 8, 4  # tiny synthetic scene: 8x8 LR, x4 scale -> 32x32 HR


def _make_loaded_subset(subset_name: str = "spot", n=N, h=H, scale=SCALE):
    rng = np.random.default_rng(42)
    l2a = rng.integers(1, 8000, size=(n, 12, h, h)).astype("uint16")
    l1c = rng.integers(1, 8000, size=(n, 13, h, h)).astype("uint16")
    hr = rng.integers(1, 8000, size=(n, 4, h * scale, h * scale)).astype("uint16")
    hrharm = rng.integers(1, 8000, size=(n, 4, h * scale, h * scale)).astype("uint16")

    metadata = pd.DataFrame(
        {
            "lr_file": [f"ROI_{i:04d}__lrfile" for i in range(n)],
            "hr_file": [f"ROI_{i:04d}__hrfile" for i in range(n)],
            "roi": [f"ROI_{i:04d}" for i in range(n)],
            "lr_gee_id": [f"gee_id_{i}" for i in range(n)],
            "reflectance": [0.1] * n,
            "spectral": [0.2] * n,
            "spatial": [0.05] * n,
            "crs": ["EPSG:32630"] * n,
            "affine": [
                f"{2.5},0.0,720285.0,0.0,-{2.5},4375125.0" for _ in range(n)
            ],
        }
    )

    return {
        "L2A": l2a,
        "L1C": l1c,
        "HR": hr,
        "HRharm": hrharm,
        "metadata": metadata,
        "hr": hr,  # undocumented duplicate keys the real dataset also carries
        "hr_harm": hrharm,
    }


# ---------------------------------------------------------------------------
# extract_sample -- core logic
# ---------------------------------------------------------------------------

def test_extracted_lr_has_four_rgbn_bands_at_native_resolution():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    assert sample.lr_reflectance.shape == (4, H, H)


def test_extracted_hr_matches_scale_factor():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    assert sample.hr_reflectance.shape == (4, H * SCALE, H * SCALE)
    assert sample.scale_factor == SCALE


def test_band_extraction_uses_the_documented_l2a_index_table():
    # L2A_BAND_ORDER is sourced from the opensr-test HF dataset card's own
    # published "L2A Index" table -- verify our RGBN indices match it
    # (B04=3, B03=2, B02=1, B08=7), and that extract_sample actually pulls
    # those exact source channels (not some other arbitrary permutation).
    assert L2A_BAND_ORDER.index("B04") == 3
    assert L2A_BAND_ORDER.index("B03") == 2
    assert L2A_BAND_ORDER.index("B02") == 1
    assert L2A_BAND_ORDER.index("B08") == 7

    loaded = _make_loaded_subset()
    l2a_raw = loaded["L2A"][0]  # (12, H, H), uint16 raw DN
    sample = extract_sample(loaded, subset="spot", sample_index=0)

    expected_reflectance = (l2a_raw[[3, 2, 1, 7]].astype("float32") / 10_000.0)
    assert torch.allclose(sample.lr_reflectance, torch.from_numpy(expected_reflectance), atol=1e-6)


def test_rgbn_band_names_match_frame_preprocessing_convention():
    from frame.preprocessing import RGBN_BANDS

    assert RGBN_BAND_NAMES == RGBN_BANDS


def test_hr_variant_defaults_to_harmonized():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    assert sample.hr_variant == "HRharm"
    expected = torch.from_numpy(loaded["HRharm"][0].astype("float32") / 10_000.0)
    assert torch.allclose(sample.hr_reflectance, expected, atol=1e-6)


def test_hr_variant_can_be_set_to_raw_hr():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0, hr_variant="HR")
    assert sample.hr_variant == "HR"
    expected = torch.from_numpy(loaded["HR"][0].astype("float32") / 10_000.0)
    assert torch.allclose(sample.hr_reflectance, expected, atol=1e-6)


def test_sample_records_subset_and_index_and_roi():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=2)
    assert sample.subset == "spot"
    assert sample.sample_index == 2
    assert sample.roi_id == "ROI_0002"


def test_geospatial_metadata_is_parsed_from_the_affine_column():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    assert sample.hr_metadata.crs == "EPSG:32630"
    assert sample.hr_metadata.transform == (2.5, 0.0, 720285.0, 0.0, -2.5, 4375125.0)
    assert sample.hr_metadata.resolution_m == pytest.approx(2.5)


def test_lr_geospatial_metadata_is_derived_by_scaling_the_hr_transform():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    # LR pixel size = HR pixel size * scale_factor; origin unchanged
    assert sample.lr_metadata.transform[0] == pytest.approx(2.5 * SCALE)
    assert sample.lr_metadata.transform[4] == pytest.approx(-2.5 * SCALE)
    assert sample.lr_metadata.transform[2] == pytest.approx(720285.0)
    assert sample.lr_metadata.transform[5] == pytest.approx(4375125.0)
    assert sample.lr_metadata.resolution_m == pytest.approx(10.0)


def test_l2a_band_order_citation_is_recorded():
    loaded = _make_loaded_subset()
    sample = extract_sample(loaded, subset="spot", sample_index=0)
    assert "isp-uv-es/opensr-test" in sample.l2a_band_order_source


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_unsupported_subset_is_rejected():
    loaded = _make_loaded_subset()
    with pytest.raises(UnsupportedSubsetError):
        extract_sample(loaded, subset="venus", sample_index=0)


def test_naip_is_not_yet_supported():
    # NAIP's LR grid (121x121) does not match our proven 128x128 patch
    # convention -- explicitly out of Phase 4 scope, not silently handled.
    loaded = _make_loaded_subset()
    with pytest.raises(UnsupportedSubsetError):
        extract_sample(loaded, subset="naip", sample_index=0)


def test_sample_index_out_of_range_is_rejected():
    loaded = _make_loaded_subset(n=3)
    with pytest.raises(SampleIndexError):
        extract_sample(loaded, subset="spot", sample_index=3)


def test_negative_sample_index_is_rejected():
    loaded = _make_loaded_subset(n=3)
    with pytest.raises(SampleIndexError):
        extract_sample(loaded, subset="spot", sample_index=-1)


def test_unexpected_l2a_band_count_is_rejected():
    loaded = _make_loaded_subset()
    loaded["L2A"] = loaded["L2A"][:, :10]  # simulate a format change upstream
    with pytest.raises(BenchmarkFormatError):
        extract_sample(loaded, subset="spot", sample_index=0)


def test_hr_shape_not_matching_scale_factor_is_rejected():
    loaded = _make_loaded_subset()
    loaded["HRharm"] = loaded["HRharm"][:, :, :-1, :-1]  # corrupt the HR shape
    with pytest.raises(BenchmarkFormatError):
        extract_sample(loaded, subset="spot", sample_index=0)


def test_supported_subsets_are_exactly_the_128x128_native_three():
    assert SUPPORTED_SUBSETS == ("spot", "spain_crops", "spain_urban")
