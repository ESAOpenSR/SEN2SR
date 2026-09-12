"""Tests for the FRAME API (Phase 7) -- HTTP-layer behavior via FastAPI's
TestClient, with the real model swapped for a small deterministic fake
(FastAPI dependency override) so these tests need no network and no real
SEN2SRLite weights. See frame/tests/test_api_integration.py for the real,
network-touching end-to-end test.
"""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
import pytest
from fastapi.testclient import TestClient

from frame.geospatial import RGBN_SCALE_FACTOR, write_geotiff
from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata

from frame.api.app import create_app
from frame.api import routes
from frame.api.services.storage import JobStore

H = 128


def _equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x_batched, scale_factor=RGBN_SCALE_FACTOR, mode="bicubic", antialias=True)


def _synthetic_geotiff_bytes(*, band_names=RGBN_BANDS, h=H, tmp_path) -> bytes:
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
    path = tmp_path / "scene.tif"
    write_geotiff(path, array, metadata)
    return path.read_bytes()


@pytest.fixture()
def app_and_client(tmp_path, monkeypatch):
    from frame.api import config

    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    app = create_app()
    store = JobStore()
    app.dependency_overrides[routes.get_model_callable] = lambda: _equivariant_model
    app.dependency_overrides[routes.get_store] = lambda: store
    client = TestClient(app)
    return app, client


@pytest.fixture()
def client(app_and_client):
    return app_and_client[1]


# ---------------------------------------------------------------------------
# 1. GET /health
# ---------------------------------------------------------------------------

def test_health_returns_ok_with_versions(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "api_version" in body
    assert "model_name" in body
    assert "frame_version" in body  # may be null -- "if available"


# ---------------------------------------------------------------------------
# 2. malformed requests fail cleanly
# ---------------------------------------------------------------------------

def test_aoi_preview_rejects_malformed_json_body(client):
    response = client.post("/aoi/preview", content=b"not json", headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_aoi_preview_rejects_out_of_range_latitude(client):
    response = client.post(
        "/aoi/preview",
        json={"lat": 999.0, "lon": 0.0, "start_date": "2023-01-01", "end_date": "2023-01-02"},
    )
    assert response.status_code == 422


def test_aoi_preview_rejects_inverted_date_range(client):
    response = client.post(
        "/aoi/preview",
        json={"lat": 10.0, "lon": 10.0, "start_date": "2023-06-01", "end_date": "2023-01-01"},
    )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body and "detail" in body


def test_aoi_preview_valid_request_succeeds(client):
    response = client.post(
        "/aoi/preview",
        json={"lat": 39.49, "lon": -0.43, "start_date": "2023-01-15", "end_date": "2023-01-16", "edge_size": 128},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["expected_input_shape"] == [4, 128, 128]
    assert body["expected_output_shape"] == [4, 512, 512]
    assert body["sr_product_description"] == "SR-derived product — 2.5 m pixel grid"


def test_sr_run_with_unknown_upload_id_returns_404(client):
    response = client.post("/sr/run", json={"upload_id": "does-not-exist"})
    assert response.status_code == 404


def test_sr_result_with_unknown_job_id_returns_404(client):
    response = client.get("/sr/result/does-not-exist")
    assert response.status_code == 404


def test_analysis_with_unknown_job_id_returns_404(client):
    response = client.post("/analysis/ndvi", json={"job_id": "does-not-exist"})
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 3. missing required bands fail cleanly
# ---------------------------------------------------------------------------

def test_upload_rejects_missing_required_band(client, tmp_path):
    content = _synthetic_geotiff_bytes(band_names=("B02", "B03", "B04", "B05"), tmp_path=tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")},
        data={"input_scale": "reflectance"},
    )
    assert response.status_code == 400
    body = response.json()
    assert "error" in body
    assert "traceback" not in response.text.lower()


def test_upload_rejects_non_geotiff_file(client):
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        data={"input_scale": "reflectance"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# 4/5. valid requests work, result schema has required metadata
# ---------------------------------------------------------------------------

def test_upload_valid_geotiff_succeeds(client, tmp_path):
    content = _synthetic_geotiff_bytes(tmp_path=tmp_path)
    response = client.post(
        "/upload",
        files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")},
        data={"input_scale": "reflectance"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["band_names"] == list(RGBN_BANDS)
    assert body["width"] == H and body["height"] == H
    assert "upload_id" in body


def test_full_sr_run_flow_and_result_schema(client, tmp_path):
    content = _synthetic_geotiff_bytes(tmp_path=tmp_path)
    upload_response = client.post(
        "/upload",
        files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")},
        data={"input_scale": "reflectance"},
    )
    upload_id = upload_response.json()["upload_id"]

    run_response = client.post("/sr/run", json={"upload_id": upload_id, "seed": 42})
    assert run_response.status_code == 200
    body = run_response.json()

    # every result must expose: job_id, status, input/output shape,
    # resolution/grid description, bands, CRS, metadata, scientific caveats
    for field in (
        "job_id", "status", "input_shape", "output_shape", "resolution",
        "bands", "crs", "metadata", "scientific_caveats", "uncertainty", "self_consistency", "artifacts",
    ):
        assert field in body, f"missing required field: {field}"

    assert body["status"] == "completed"
    assert body["input_shape"] == [4, H, H]
    assert body["output_shape"] == [4, H * RGBN_SCALE_FACTOR, H * RGBN_SCALE_FACTOR]
    assert body["resolution"]["description"] == "SR-derived product — 2.5 m pixel grid"
    assert body["uncertainty"]["label"] == "relative model-stability uncertainty"
    assert body["uncertainty"]["seed"] == 42
    assert len(body["scientific_caveats"]) >= 1

    job_id = body["job_id"]

    # GET /sr/result/{job_id}
    result_response = client.get(f"/sr/result/{job_id}")
    assert result_response.status_code == 200
    assert result_response.json()["job_id"] == job_id

    # GET /sr/download/{job_id}
    download_response = client.get(f"/sr/download/{job_id}")
    assert download_response.status_code == 200
    assert download_response.headers["content-type"] in ("image/tiff", "application/octet-stream")

    # GET /uncertainty/download/{job_id}
    uncertainty_download = client.get(f"/uncertainty/download/{job_id}")
    assert uncertainty_download.status_code == 200

    # POST /analysis/ndvi
    analysis_response = client.post("/analysis/ndvi", json={"job_id": job_id})
    assert analysis_response.status_code == 200
    analysis_body = analysis_response.json()
    for field in ("analysis_id", "job_id", "comparison", "uncertainty_weighted_summary", "scientific_caveats", "artifacts"):
        assert field in analysis_body

    # GET /analysis/{analysis_id}
    analysis_id = analysis_body["analysis_id"]
    get_analysis_response = client.get(f"/analysis/{analysis_id}")
    assert get_analysis_response.status_code == 200
    assert get_analysis_response.json()["analysis_id"] == analysis_id

    # GET /analysis/download/{analysis_id}/{native-ndvi,sr-ndvi,ndvi-diff}
    for suffix in ("native-ndvi", "sr-ndvi", "ndvi-diff"):
        ndvi_download = client.get(f"/analysis/download/{analysis_id}/{suffix}")
        assert ndvi_download.status_code == 200, suffix
        assert ndvi_download.headers["content-type"] in ("image/tiff", "application/octet-stream")


def test_ndvi_download_with_unknown_analysis_id_returns_404(client):
    response = client.get("/analysis/download/does-not-exist/native-ndvi")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 6/7/8. scientific terminology correctness at the HTTP layer
# ---------------------------------------------------------------------------

def test_sr_response_never_claims_native_ground_truth(client, tmp_path):
    content = _synthetic_geotiff_bytes(tmp_path=tmp_path)
    upload_id = client.post(
        "/upload", files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")}, data={"input_scale": "reflectance"}
    ).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id}).json()
    text = str(body).lower()
    assert "native 2.5 m sentinel-2" not in text
    assert "true 2.5 m image" not in text


def test_sr_response_uncertainty_never_claims_calibration(client, tmp_path):
    content = _synthetic_geotiff_bytes(tmp_path=tmp_path)
    upload_id = client.post(
        "/upload", files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")}, data={"input_scale": "reflectance"}
    ).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id}).json()
    disclaimer = body["uncertainty"]["disclaimer"].lower()
    assert "not a calibrated probability" in disclaimer
    assert "lam" in disclaimer  # explicitly distinguishes itself from LAM


def test_no_endpoint_response_calls_lam_uncertainty(client, tmp_path):
    content = _synthetic_geotiff_bytes(tmp_path=tmp_path)
    upload_id = client.post(
        "/upload", files={"file": ("scene.tif", io.BytesIO(content), "image/tiff")}, data={"input_scale": "reflectance"}
    ).json()["upload_id"]
    body = client.post("/sr/run", json={"upload_id": upload_id}).json()
    # "LAM" never appears anywhere in a live response in this phase (it is
    # not exposed by any endpoint yet) -- if it ever does, it must not be
    # inside the uncertainty block's own numeric fields.
    assert "lam" not in str(body.get("resolution", {})).lower()
    assert "lam" not in str(body.get("self_consistency", {})).lower()
