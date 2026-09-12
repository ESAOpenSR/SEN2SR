"""HTTP routes for the FRAME API (Phase 7).

Every route is thin HTTP glue: parse the request, call
`frame.api.services.pipeline` (which itself only orchestrates the existing,
already-tested `frame.*` packages), and shape the result into a response
schema. No pipeline logic lives here. Errors are never caught locally --
they propagate to `frame.api.app`'s single generic exception handler, which
maps every `frame.api.errors`/`frame.*` exception to the right HTTP status
via `frame.api.errors.status_code_for` and never leaks a traceback.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from frame.api import config, schemas
from frame.api.errors import APIError, UnsupportedFileError
from frame.api.services import model as model_service
from frame.api.services import pipeline
from frame.api.services.storage import JobStore

router = APIRouter()

# A single process-lifetime store -- see frame.api.services.storage's own
# docstring for why this is an in-memory registry, not a database. Tests
# override `get_store` (FastAPI dependency injection) to get an isolated,
# empty store per test.
_default_store = JobStore()


def get_store() -> JobStore:
    return _default_store


def get_device() -> str:
    return model_service.resolve_device(config.DEVICE)


def get_model_callable(device: str = Depends(get_device)) -> Callable:
    """The cached, compiled SEN2SRLite model -- overridden with a small fake
    callable in every non-integration test (see frame/tests/test_api.py)."""
    return model_service.get_model(device)


SUPPORTED_UPLOAD_EXTENSIONS = (".tif", ".tiff")
REQUIRED_BANDS = ["B04", "B03", "B02", "B08"]


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=schemas.HealthResponse)
def health() -> schemas.HealthResponse:
    import frame

    frame_version = getattr(frame, "__version__", None)
    return schemas.HealthResponse(
        status="ok", frame_version=frame_version, api_version=config.API_VERSION, model_name=config.MODEL_NAME
    )


# ---------------------------------------------------------------------------
# POST /aoi/preview
# ---------------------------------------------------------------------------

@router.post("/aoi/preview", response_model=schemas.AOIPreviewResponse)
def aoi_preview(request: schemas.AOIPreviewRequest) -> schemas.AOIPreviewResponse:
    if request.start_date > request.end_date:
        raise APIError(f"start_date ({request.start_date}) must be <= end_date ({request.end_date}).")
    if request.bands != REQUIRED_BANDS:
        raise APIError(
            f"This prototype only supports bands {REQUIRED_BANDS} (in that order), got {request.bands}."
        )
    scale_factor = 4
    return schemas.AOIPreviewResponse(
        valid=True,
        coordinates={"lat": request.lat, "lon": request.lon},
        date_window={"start": request.start_date, "end": request.end_date},
        edge_size=request.edge_size,
        bands=request.bands,
        expected_input_shape=[len(request.bands), request.edge_size, request.edge_size],
        expected_output_shape=[len(request.bands), request.edge_size * scale_factor, request.edge_size * scale_factor],
        native_resolution_m=10.0,
        sr_resolution_m=10.0 / scale_factor,
    )


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=schemas.UploadResponse)
def upload(
    file: UploadFile = File(...),
    input_scale: str = Form("raw_digital_number"),
    store: JobStore = Depends(get_store),
) -> schemas.UploadResponse:
    filename = file.filename or "upload"
    if not filename.lower().endswith(SUPPORTED_UPLOAD_EXTENSIONS):
        raise UnsupportedFileError(
            f"Unsupported file type for {filename!r}; expected a GeoTIFF ({', '.join(SUPPORTED_UPLOAD_EXTENSIONS)})."
        )

    config.ensure_workspace_dirs()
    dest = config.WORKSPACE_DIR / "uploads" / f"{uuid.uuid4().hex}_{filename}"
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    record = pipeline.process_upload(store, dest, filename=filename, input_scale=input_scale)
    return schemas.UploadResponse(
        upload_id=record.upload_id,
        filename=record.filename,
        valid=True,
        band_names=list(record.band_names),
        width=record.width,
        height=record.height,
        crs=record.crs,
        resolution_m=record.resolution_m,
        input_scale=record.input_scale,
        validation_messages=["Upload accepted: required bands and shape present, CRS valid."],
    )


# ---------------------------------------------------------------------------
# POST /sr/run
# ---------------------------------------------------------------------------

@router.post("/sr/run", response_model=schemas.SRResultResponse)
def sr_run(
    request: schemas.SRRunRequest,
    store: JobStore = Depends(get_store),
    model: Callable = Depends(get_model_callable),
    device: str = Depends(get_device),
) -> schemas.SRResultResponse:
    config.ensure_workspace_dirs()
    seed = request.seed if request.seed is not None else config.UNCERTAINTY_SEED
    job = pipeline.run_sr_job(
        store, request.upload_id, model=model, device=device, seed=seed, workspace_dir=config.WORKSPACE_DIR / "jobs"
    )
    return schemas.SRResultResponse(**job.result)


# ---------------------------------------------------------------------------
# POST /analysis/ndvi
# ---------------------------------------------------------------------------

@router.post("/analysis/ndvi", response_model=schemas.NDVIAnalysisResponse)
def analysis_ndvi(request: schemas.NDVIAnalysisRequest, store: JobStore = Depends(get_store)) -> schemas.NDVIAnalysisResponse:
    analysis = pipeline.run_ndvi_analysis_job(store, request.job_id)
    return schemas.NDVIAnalysisResponse(**analysis.result)


# ---------------------------------------------------------------------------
# GET /sr/result/{job_id}
# ---------------------------------------------------------------------------

@router.get("/sr/result/{job_id}", response_model=schemas.SRResultResponse)
def get_sr_result(job_id: str, store: JobStore = Depends(get_store)) -> schemas.SRResultResponse:
    job = store.get_job(job_id)
    return schemas.SRResultResponse(**job.result)


# ---------------------------------------------------------------------------
# GET /sr/download/{job_id}
# ---------------------------------------------------------------------------

@router.get("/sr/download/{job_id}")
def download_sr(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get_job(job_id)
    return FileResponse(job.sr_mean_path, media_type="image/tiff", filename=f"sr_{job_id}.tif")


# ---------------------------------------------------------------------------
# GET /uncertainty/download/{job_id}
# ---------------------------------------------------------------------------

@router.get("/uncertainty/download/{job_id}")
def download_uncertainty(job_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    job = store.get_job(job_id)
    return FileResponse(job.sr_std_path, media_type="image/tiff", filename=f"uncertainty_{job_id}.tif")


# ---------------------------------------------------------------------------
# GET /analysis/{analysis_id}
# ---------------------------------------------------------------------------

@router.get("/analysis/{analysis_id}", response_model=schemas.NDVIAnalysisResponse)
def get_analysis(analysis_id: str, store: JobStore = Depends(get_store)) -> schemas.NDVIAnalysisResponse:
    analysis = store.get_analysis(analysis_id)
    return schemas.NDVIAnalysisResponse(**analysis.result)


# ---------------------------------------------------------------------------
# GET /analysis/download/{analysis_id}/{native-ndvi,sr-ndvi,ndvi-diff}
#
# Phase 8 (frontend) addition: the NDVI view needs the actual rasters, not
# just the scalar summary above -- frame.analysis already computes them
# (frame.api.services.pipeline.run_ndvi_analysis_job writes them out; no
# new scientific computation was added for this). Single-band GeoTIFFs,
# same download shape as /sr/download and /uncertainty/download.
# ---------------------------------------------------------------------------

@router.get("/analysis/download/{analysis_id}/native-ndvi")
def download_native_ndvi(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.native_ndvi_path, media_type="image/tiff", filename=f"ndvi_native_{analysis_id}.tif")


@router.get("/analysis/download/{analysis_id}/sr-ndvi")
def download_sr_ndvi(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.sr_ndvi_path, media_type="image/tiff", filename=f"ndvi_sr_{analysis_id}.tif")


@router.get("/analysis/download/{analysis_id}/ndvi-diff")
def download_ndvi_diff(analysis_id: str, store: JobStore = Depends(get_store)) -> FileResponse:
    analysis = store.get_analysis(analysis_id)
    return FileResponse(analysis.ndvi_diff_path, media_type="image/tiff", filename=f"ndvi_diff_{analysis_id}.tif")
