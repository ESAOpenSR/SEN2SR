"""Integration test: the uncertainty raster produced by frame.uncertainty
stays exactly co-registered with the SR raster it describes, by REUSING
frame.geospatial (Phase 2) rather than frame.uncertainty duplicating any
georeferencing logic of its own.

frame.uncertainty itself never imports frame.geospatial or writes files --
this test proves the *pattern* experiments/uncertainty/run_experiment.py
uses actually holds: the same RasterMetadata that describes the SR output
also correctly describes the uncertainty output (same CRS, transform,
bounds, width, height), because both come from the same
`derive_output_metadata` call with the same scale_factor.
"""

import numpy as np
import torch
import torch.nn.functional as F

from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, read_geotiff, write_geotiff
from frame.preprocessing.metadata import RasterMetadata
from frame.uncertainty.report import run_stochastic_uncertainty

BANDS = ("B04", "B03", "B02", "B08")


def _equivariant_model(x_batched):
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _input_metadata():
    return RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375045.0, 720365.0, 4375125.0),  # consistent with 8x8 @ 10m
        resolution_m=10.0,
        width=8,
        height=8,
        band_names=BANDS,
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant="SEN2SRLite/NonReference_RGBN_x4",
    )


def test_uncertainty_raster_shares_sr_rasters_geospatial_metadata(tmp_path):
    torch.manual_seed(0)
    x = torch.rand(4, 8, 8) * 0.4 + 0.1

    input_metadata = _input_metadata()
    output_metadata = derive_output_metadata(
        input_metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=input_metadata.band_names
    )

    result = run_stochastic_uncertainty(_equivariant_model, x, seed=42, band_names=BANDS)

    # the SR mean prediction and the uncertainty (std) map must both match
    # the SAME output_metadata's declared width/height exactly
    assert result.mean_prediction.shape[-2:] == (output_metadata.height, output_metadata.width)
    assert result.std_prediction.shape[-2:] == (output_metadata.height, output_metadata.width)

    sr_path = tmp_path / "sr.tif"
    uncertainty_path = tmp_path / "uncertainty_std.tif"
    write_geotiff(sr_path, result.mean_prediction.numpy(), output_metadata)
    write_geotiff(uncertainty_path, result.std_prediction.numpy(), output_metadata)

    _, sr_read_meta = read_geotiff(sr_path)
    _, uncertainty_read_meta = read_geotiff(uncertainty_path)

    assert sr_read_meta.crs == uncertainty_read_meta.crs
    assert sr_read_meta.transform == uncertainty_read_meta.transform
    assert sr_read_meta.bounds == uncertainty_read_meta.bounds
    assert sr_read_meta.width == uncertainty_read_meta.width
    assert sr_read_meta.height == uncertainty_read_meta.height
    assert sr_read_meta.resolution_m == uncertainty_read_meta.resolution_m
