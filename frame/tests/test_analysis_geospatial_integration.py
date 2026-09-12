"""Integration test: frame.analysis's native-grid and SR-grid NDVI results
stay aligned with the geospatial metadata frame.geospatial (Phase 2) would
derive for the same LR/SR tensors -- same pattern as
frame/tests/test_uncertainty_geospatial_integration.py (Phase 5).

frame.analysis itself never imports frame.geospatial -- this test proves
the *pattern* experiments/analysis/run_experiment.py uses actually holds.
"""

import torch

from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata
from frame.preprocessing.metadata import RasterMetadata
from frame.analysis.report import run_ndvi_analysis

BANDS = ("B04", "B03", "B02", "B08")
H = 8


def _input_metadata():
    return RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375045.0, 720365.0, 4375125.0),
        resolution_m=10.0,
        width=H,
        height=H,
        band_names=BANDS,
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )


def test_native_and_sr_ndvi_grids_match_frame_geospatials_own_derived_resolution_and_size():
    torch.manual_seed(0)
    lr = torch.rand(4, H, H) * 0.4 + 0.1
    sr_mean = torch.rand(4, H * RGBN_SCALE_FACTOR, H * RGBN_SCALE_FACTOR) * 0.4 + 0.1
    sr_std = torch.rand(4, H * RGBN_SCALE_FACTOR, H * RGBN_SCALE_FACTOR) * 0.01

    input_metadata = _input_metadata()
    output_metadata = derive_output_metadata(
        input_metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=input_metadata.band_names
    )

    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=RGBN_SCALE_FACTOR)

    # native NDVI grid matches the INPUT metadata exactly
    assert report.native_ndvi.ndvi.shape == (input_metadata.height, input_metadata.width)
    assert report.native_ndvi.resolution_m == input_metadata.resolution_m

    # SR NDVI grid matches the geospatially-derived OUTPUT metadata exactly
    assert report.sr_ndvi.ndvi.shape == (output_metadata.height, output_metadata.width)
    assert report.sr_ndvi.resolution_m == output_metadata.resolution_m

    # the downsampled-SR-NDVI-on-native-grid and the uncertainty native-grid
    # map both land back on the INPUT metadata's own grid size
    assert report.ndvi_comparison.sr_ndvi_downsampled.shape == (input_metadata.height, input_metadata.width)
    assert report.uncertainty_overall_native_grid.shape == (input_metadata.height, input_metadata.width)
