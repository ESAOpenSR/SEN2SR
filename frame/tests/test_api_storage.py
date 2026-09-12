"""Tests for frame.api.services.storage -- the in-memory upload/job/analysis
registry backing the prototype API (Phase 7). Explicitly not a production
job queue -- an in-process dict, appropriate for a synchronous demo backend.
"""

from pathlib import Path

import pytest

from frame.api.errors import JobNotFoundError, UploadNotFoundError
from frame.api.services.storage import JobStore, SRJobRecord, UploadRecord
from frame.preprocessing.metadata import RasterMetadata


def _upload_record(upload_id="up-1"):
    return UploadRecord(
        upload_id=upload_id,
        filename="scene.tif",
        file_path=Path("/tmp/scene.tif"),
        band_names=("B04", "B03", "B02", "B08"),
        width=128,
        height=128,
        crs="EPSG:32630",
        resolution_m=10.0,
        input_scale="raw_digital_number",
        created_at="2026-01-01T00:00:00Z",
    )


def test_put_and_get_upload_round_trips():
    store = JobStore()
    record = _upload_record()
    store.put_upload(record)
    assert store.get_upload("up-1") is record


def test_get_missing_upload_raises_upload_not_found():
    store = JobStore()
    with pytest.raises(UploadNotFoundError):
        store.get_upload("does-not-exist")


def test_put_and_get_job_round_trips():
    store = JobStore()
    job = SRJobRecord(
        job_id="job-1",
        status="completed",
        upload_id="up-1",
        input_tensor_path=Path("/tmp/lr.pt"),
        sr_mean_path=Path("/tmp/sr_mean.tif"),
        sr_std_path=Path("/tmp/sr_std.tif"),
        sr_mean_tensor_path=Path("/tmp/sr_mean.pt"),
        sr_std_tensor_path=Path("/tmp/sr_std.pt"),
        input_metadata=RasterMetadata.unknown(
            band_names=("B04", "B03", "B02", "B08"), width=128, height=128, resolution_m=10.0, sr_variant="lr"
        ),
        output_metadata=RasterMetadata.unknown(
            band_names=("B04", "B03", "B02", "B08"), width=512, height=512, resolution_m=2.5, sr_variant="sr"
        ),
        result={"foo": "bar"},
        error=None,
        created_at="2026-01-01T00:00:00Z",
    )
    store.put_job(job)
    assert store.get_job("job-1") is job


def test_get_missing_job_raises_job_not_found():
    store = JobStore()
    with pytest.raises(JobNotFoundError):
        store.get_job("does-not-exist")


def test_stores_are_independent_per_instance():
    a = JobStore()
    b = JobStore()
    a.put_upload(_upload_record("only-in-a"))
    with pytest.raises(UploadNotFoundError):
        b.get_upload("only-in-a")


def test_generate_id_produces_unique_values():
    store = JobStore()
    ids = {store.generate_id() for _ in range(100)}
    assert len(ids) == 100
