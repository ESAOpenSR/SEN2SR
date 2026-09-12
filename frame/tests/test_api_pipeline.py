"""Tests for frame.api.services.pipeline -- orchestration of the existing,
already-tested frame.preprocessing / frame.geospatial / frame.uncertainty /
frame.consistency / frame.analysis modules behind the API.

Uses a synthetic GeoTIFF fixture (written via the real frame.geospatial.
write_geotiff, Phase 2) and a small deterministic fake "model" callable
(same pattern as frame/tests/test_uncertainty_ensemble.py) -- no network,
no real SEN2SRLite weights.
"""

import inspect

import numpy as np
import torch
import torch.nn.functional as F
import pytest

from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.errors import UnsupportedBandsError
from frame.preprocessing.metadata import RasterMetadata
from frame.api.services import pipeline
from frame.api.services.storage import JobStore

H = 128  # the one proven patch size


def _equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _write_synthetic_geotiff(path, *, band_names=RGBN_BANDS, h=H):
    rng = np.random.default_rng(0)
    array = (rng.random((len(band_names), h, h)).astype("float32") * 0.4) + 0.1
    metadata = RasterMetadata(
        crs="EPSG:32630",
        transform=(10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0),
        bounds=(720285.0, 4375125.0 - h * 10.0, 720285.0 + h * 10.0, 4375125.0),
        resolution_m=10.0,
        width=h,
        height=h,
        band_names=tuple(band_names),
        acquisition_timestamp="2023-01-15T10:54:11.024000",
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant=None,
    )
    write_geotiff(path, array, metadata)
    return array, metadata


# ---------------------------------------------------------------------------
# process_upload
# ---------------------------------------------------------------------------

def test_process_upload_reads_a_valid_geotiff(tmp_path):
    path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(path)
    store = JobStore()
    record = pipeline.process_upload(store, path, filename="scene.tif", input_scale="reflectance")
    assert record.band_names == RGBN_BANDS
    assert record.width == H and record.height == H
    assert record.crs == "EPSG:32630"
    assert store.get_upload(record.upload_id) is record


def test_process_upload_rejects_missing_required_band(tmp_path):
    path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(path, band_names=("B02", "B03", "B04", "B05"))  # no B08
    store = JobStore()
    with pytest.raises(UnsupportedBandsError):
        pipeline.process_upload(store, path, filename="scene.tif", input_scale="reflectance")


# ---------------------------------------------------------------------------
# run_sr_job
# ---------------------------------------------------------------------------

def test_run_sr_job_produces_correct_shapes_and_artifacts(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")

    workspace = tmp_path / "job_workspace"
    job = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=42, workspace_dir=workspace
    )
    assert job.status == "completed"
    assert job.sr_mean_path.exists()
    assert job.sr_std_path.exists()
    assert job.sr_mean_tensor_path.exists()
    assert job.sr_std_tensor_path.exists()
    assert job.input_tensor_path.exists()

    assert job.result["input_shape"] == [4, H, H]
    assert job.result["output_shape"] == [4, H * RGBN_SCALE_FACTOR, H * RGBN_SCALE_FACTOR]
    assert job.result["bands"] == list(RGBN_BANDS)
    assert job.result["crs"] == "EPSG:32630"


def test_run_sr_job_result_includes_uncertainty_and_self_consistency_and_caveats(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "ws"
    )
    assert "uncertainty" in job.result
    assert job.result["uncertainty"]["seed"] == 42
    assert job.result["uncertainty"]["n"] == 6
    assert "self_consistency" in job.result
    assert len(job.result["scientific_caveats"]) >= 1


def test_run_sr_job_is_registered_in_the_store(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "ws"
    )
    assert store.get_job(job.job_id) is job


def test_run_sr_job_seed_is_reproducible(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job_a = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=7, workspace_dir=tmp_path / "ws_a"
    )
    job_b = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=7, workspace_dir=tmp_path / "ws_b"
    )
    assert job_a.result["uncertainty"]["scalar_summary"] == job_b.result["uncertainty"]["scalar_summary"]


# ---------------------------------------------------------------------------
# run_ndvi_analysis_job
# ---------------------------------------------------------------------------

def test_run_ndvi_analysis_job_produces_expected_result_fields(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "ws"
    )
    analysis = pipeline.run_ndvi_analysis_job(store, job.job_id)
    assert analysis.job_id == job.job_id
    assert "comparison" in analysis.result
    assert "uncertainty_weighted_summary" in analysis.result
    assert store.get_analysis(analysis.analysis_id) is analysis


def test_run_ndvi_analysis_job_writes_ndvi_geotiffs(tmp_path):
    upload_path = tmp_path / "scene.tif"
    _write_synthetic_geotiff(upload_path)
    store = JobStore()
    upload = pipeline.process_upload(store, upload_path, filename="scene.tif", input_scale="reflectance")
    job = pipeline.run_sr_job(
        store, upload.upload_id, model=_equivariant_model, device="cpu", seed=42, workspace_dir=tmp_path / "ws"
    )
    analysis = pipeline.run_ndvi_analysis_job(store, job.job_id)

    assert analysis.native_ndvi_path.exists()
    assert analysis.sr_ndvi_path.exists()
    assert analysis.ndvi_diff_path.exists()
    assert analysis.result["artifacts"] == {
        "native_ndvi_geotiff": str(analysis.native_ndvi_path),
        "sr_ndvi_geotiff": str(analysis.sr_ndvi_path),
        "ndvi_diff_geotiff": str(analysis.ndvi_diff_path),
    }

    from frame.geospatial import read_geotiff

    native_array, native_meta = read_geotiff(analysis.native_ndvi_path)
    assert native_array.shape == (1, H, H)
    assert native_meta.band_names == ("NDVI",)

    sr_array, sr_meta = read_geotiff(analysis.sr_ndvi_path)
    assert sr_array.shape == (1, H * RGBN_SCALE_FACTOR, H * RGBN_SCALE_FACTOR)
    assert sr_meta.band_names == ("NDVI",)

    diff_array, _ = read_geotiff(analysis.ndvi_diff_path)
    assert diff_array.shape == (1, H, H)


# ---------------------------------------------------------------------------
# Orchestration, not reimplementation (Phase 7 test requirement 9)
# ---------------------------------------------------------------------------

def test_pipeline_module_imports_the_real_frame_functions_rather_than_reimplementing_them():
    source = inspect.getsource(pipeline)
    for expected_import in [
        "from frame.preprocessing import",
        "from frame.geospatial import",
        "from frame.uncertainty import",
        "from frame.consistency import",
        "from frame.analysis import",
    ]:
        assert expected_import in source, f"pipeline.py does not import via {expected_import!r}"
    # and it must never import sen2sr/mlstac directly -- only frame.* and,
    # for the model call itself, whatever callable the caller injects
    assert "import sen2sr" not in source
    assert "import mlstac" not in source
