"""INTEGRATION test for the FRAME API (Phase 7).

Unlike frame/tests/test_api.py (which overrides the model dependency with a
small deterministic fake so it needs no network and no real weights), this
test runs the FULL API stack against the REAL, cached
SEN2SRLite/NonReference_RGBN_x4 model -- the same weights cache every prior
phase's experiments already populated at
``$SEN2SR_BASELINE_WEIGHTS_DIR`` (default ``~/.cache/sen2sr_baseline/SEN2SRLite_RGBN``).
If that cache is missing, this test is skipped rather than triggering a
fresh download. It is also excluded from a normal `pytest frame/tests/`
run regardless of cache state -- `pyproject.toml` sets
`addopts = "-m 'not integration'"`, so this module's `integration` marker
(`pytestmark` below) keeps it out of the default run entirely.

The uploaded scene is reconstructed from the real, deterministic Baseline 0
artifact (`experiments/baseline/outputs/input_tensor.pt` -- the actual
fetched Sentinel-2 L2A raw-digital-number tensor for the fixed AOI/date
window documented in `experiments/baseline/metadata/run_metadata.json` and
`experiments/baseline_geoexport/metadata/run_metadata.json`), not a
synthetic array -- this is "the existing deterministic Baseline 0
scene/artifacts" the phase instructions call for. No new network fetch of
Sentinel-2 imagery happens here.

Run explicitly with:
    sen2sr_venv/bin/python -m pytest frame/tests/ -m integration -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from frame.geospatial import write_geotiff
from frame.preprocessing.metadata import RasterMetadata

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_INPUT_TENSOR = REPO_ROOT / "experiments" / "baseline" / "outputs" / "input_tensor.pt"

# Real, fixed AOI/scene identity from experiments/baseline_geoexport/metadata/run_metadata.json
# (itself required to match experiments/baseline/run_baseline.py exactly).
BANDS = ["B04", "B03", "B02", "B08"]
CRS = "EPSG:32630"
TRANSFORM = (10.0, 0.0, 720285.0, 0.0, -10.0, 4375125.0)
BOUNDS = (720285.0, 4373845.0, 721565.0, 4375125.0)
ACQUISITION_TIMESTAMP = "2023-01-15T10:54:11.024000"

from frame.api import config as api_config  # noqa: E402


def _weights_cached() -> bool:
    return (api_config.MODEL_WEIGHTS_CACHE_DIR / "mlm.json").exists()


def _baseline_artifact_available() -> bool:
    return BASELINE_INPUT_TENSOR.exists()


skip_reason = None
if not _weights_cached():
    skip_reason = f"No cached SEN2SRLite weights at {api_config.MODEL_WEIGHTS_CACHE_DIR}; skipping rather than downloading."
elif not _baseline_artifact_available():
    skip_reason = f"Missing deterministic Baseline 0 artifact at {BASELINE_INPUT_TENSOR}; run experiments/baseline first."


@pytest.mark.skipif(skip_reason is not None, reason=str(skip_reason))
def test_real_pipeline_end_to_end_against_baseline_0_scene(tmp_path, monkeypatch):
    """Full, real (no fakes/overrides) FRAME API pipeline run: upload the
    real Baseline 0 scene, run SR with the real cached model, run NDVI
    analysis, and download both GeoTIFFs -- proving the API layer, not
    just the underlying frame.* modules, works against the real model."""
    from frame.api import config
    from frame.api.app import create_app
    from frame.api.services import model as model_service

    monkeypatch.setattr(config, "WORKSPACE_DIR", tmp_path / "workspace")
    model_service.clear_cache()

    raw_dn = torch.load(BASELINE_INPUT_TENSOR, weights_only=True).numpy()  # (4, 128, 128), raw digital numbers
    metadata = RasterMetadata(
        crs=CRS,
        transform=TRANSFORM,
        bounds=BOUNDS,
        resolution_m=10.0,
        width=raw_dn.shape[-1],
        height=raw_dn.shape[-2],
        band_names=tuple(BANDS),
        acquisition_timestamp=ACQUISITION_TIMESTAMP,
        nodata_value=0.0,
        cloud_mask_coverage=1.0,
        sr_variant=None,
    )
    scene_path = tmp_path / "baseline_0_scene.tif"
    write_geotiff(scene_path, raw_dn, metadata)

    app = create_app()
    client = TestClient(app)

    with open(scene_path, "rb") as f:
        upload_response = client.post(
            "/upload",
            files={"file": ("baseline_0_scene.tif", f, "image/tiff")},
            data={"input_scale": "raw_digital_number"},
        )
    assert upload_response.status_code == 200, upload_response.text
    upload_body = upload_response.json()
    assert upload_body["band_names"] == BANDS
    upload_id = upload_body["upload_id"]

    run_response = client.post("/sr/run", json={"upload_id": upload_id, "seed": 42})
    assert run_response.status_code == 200, run_response.text
    result = run_response.json()

    assert result["status"] == "completed"
    assert result["input_shape"] == [4, 128, 128]
    assert result["output_shape"] == [4, 512, 512]
    assert result["resolution"]["description"] == "SR-derived product — 2.5 m pixel grid"
    assert result["uncertainty"]["label"] == "relative model-stability uncertainty"
    assert result["uncertainty"]["seed"] == 42
    assert result["uncertainty"]["n"] >= 1
    assert result["crs"] == CRS
    assert len(result["scientific_caveats"]) >= 1

    job_id = result["job_id"]

    sr_download = client.get(f"/sr/download/{job_id}")
    assert sr_download.status_code == 200
    assert len(sr_download.content) > 0

    uncertainty_download = client.get(f"/uncertainty/download/{job_id}")
    assert uncertainty_download.status_code == 200
    assert len(uncertainty_download.content) > 0

    analysis_response = client.post("/analysis/ndvi", json={"job_id": job_id})
    assert analysis_response.status_code == 200, analysis_response.text
    analysis_body = analysis_response.json()
    assert analysis_body["job_id"] == job_id
    assert "comparison" in analysis_body
    assert "uncertainty_weighted_summary" in analysis_body

    model_service.clear_cache()
