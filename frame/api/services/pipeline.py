"""Pipeline orchestration for the FRAME API (Phase 7).

This module does not reimplement Phase 1-6 logic -- it calls the existing,
already-tested modules in sequence and packages their results:

    ingestion (an uploaded GeoTIFF, read via frame.geospatial.read_geotiff)
      -> preprocessing (frame.preprocessing.preprocess_rgbn)
      -> frozen SEN2SR + uncertainty in one call
         (frame.uncertainty.run_stochastic_uncertainty -- its TTA ensemble
          IS the model inference step; running it once gives both the SR
          mean prediction and the model-stability uncertainty, rather than
          calling the model a second time separately)
      -> geospatial attachment (frame.geospatial.derive_output_metadata,
         write_geotiff)
      -> self-consistency (frame.consistency.run_consistency_diagnostics)

`run_ndvi_analysis_job` similarly calls `frame.analysis.run_ndvi_analysis`
directly on a completed SR job's saved tensors.

The actual model object is always injected by the caller (`model=...`,
typically `frame.api.services.model.get_model(...)`) -- this module never
imports `sen2sr` or `mlstac` itself, keeping it testable with a small fake
callable (see frame/tests/test_api_pipeline.py) exactly like every prior
phase's own tests.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import torch

from frame.analysis import run_ndvi_analysis
from frame.consistency import run_consistency_diagnostics
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, read_geotiff, write_geotiff
from frame.preprocessing import PROVEN_PATCH_SIZE, RGBN_BANDS, preprocess_rgbn
from frame.preprocessing.validation import validate_bands, validate_shape
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty

from frame.api.schemas import SCIENTIFIC_CAVEATS, SR_PRODUCT_DESCRIPTION, UNCERTAINTY_DISCLAIMER, UNCERTAINTY_LABEL
from frame.api.services.storage import AnalysisRecord, JobStore, SRJobRecord, UploadRecord

NATIVE_RESOLUTION_M = 10.0


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

def process_upload(store: JobStore, file_path: Path, *, filename: str, input_scale: str) -> UploadRecord:
    """Read and validate an uploaded Sentinel-2 L2A GeoTIFF. Does NOT run SR.

    Raises the underlying `frame.geospatial`/`frame.preprocessing` error
    directly on invalid input (missing CRS, wrong band set, wrong shape) --
    frame.api.errors.status_code_for maps these to HTTP 400 at the route
    layer, not duplicated here.
    """
    array, metadata = read_geotiff(file_path, require_crs=True)
    validate_bands(list(metadata.band_names), list(RGBN_BANDS))
    validate_shape(array, expected_size=PROVEN_PATCH_SIZE)

    record = UploadRecord(
        upload_id=store.generate_id(),
        filename=filename,
        file_path=file_path,
        band_names=tuple(metadata.band_names),
        width=metadata.width,
        height=metadata.height,
        crs=metadata.crs,
        resolution_m=metadata.resolution_m,
        input_scale=input_scale,
        created_at=_now_iso(),
    )
    store.put_upload(record)
    return record


# ---------------------------------------------------------------------------
# POST /sr/run
# ---------------------------------------------------------------------------

def run_sr_job(
    store: JobStore,
    upload_id: str,
    *,
    model: Callable[[torch.Tensor], torch.Tensor],
    device: str,
    seed: int,
    workspace_dir: Path,
) -> SRJobRecord:
    """Run the full FRAME pipeline for a previously-uploaded input."""
    upload = store.get_upload(upload_id)

    array, geo_metadata = read_geotiff(upload.file_path, require_crs=True)
    preprocessed = preprocess_rgbn(
        array,
        band_names=list(upload.band_names),
        input_scale=upload.input_scale,
        resolution_m=geo_metadata.resolution_m,
        nodata_value=geo_metadata.nodata_value,
        crs=geo_metadata.crs,
        transform=geo_metadata.transform,
        bounds=geo_metadata.bounds,
        acquisition_timestamp=geo_metadata.acquisition_timestamp,
        require_geospatial=True,
    )

    X = preprocessed.tensor.to(device)
    t0 = time.time()
    uncertainty_result = run_stochastic_uncertainty(
        model, X, transforms=DEFAULT_TRANSFORMS, seed=seed, band_names=RGBN_BANDS
    )
    inference_seconds = time.time() - t0

    output_metadata = derive_output_metadata(
        preprocessed.metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=preprocessed.metadata.band_names
    )

    job_id = store.generate_id()
    job_dir = workspace_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    mean_prediction = uncertainty_result.mean_prediction
    std_prediction = uncertainty_result.std_prediction

    sr_mean_path = job_dir / "sr_mean.tif"
    write_geotiff(sr_mean_path, mean_prediction.numpy(), output_metadata)

    overall_std = std_prediction.mean(dim=0).numpy()
    uncertainty_band_names = tuple(f"{b}_std" for b in RGBN_BANDS) + ("overall_std",)
    uncertainty_array = np.concatenate([std_prediction.numpy(), overall_std[None]], axis=0)
    uncertainty_metadata = dataclasses.replace(output_metadata, band_names=uncertainty_band_names, nodata_value=None)
    sr_std_path = job_dir / "uncertainty.tif"
    write_geotiff(sr_std_path, uncertainty_array, uncertainty_metadata)

    input_tensor_path = job_dir / "input_tensor.pt"
    sr_mean_tensor_path = job_dir / "sr_mean_tensor.pt"
    sr_std_tensor_path = job_dir / "sr_std_tensor.pt"
    torch.save(preprocessed.tensor.cpu(), input_tensor_path)
    torch.save(mean_prediction, sr_mean_tensor_path)
    torch.save(std_prediction, sr_std_tensor_path)

    consistency = run_consistency_diagnostics(
        lr=preprocessed.tensor,
        sr=mean_prediction,
        mask=preprocessed.mask.array,
        band_names=RGBN_BANDS,
        scale_factor=RGBN_SCALE_FACTOR,
    )

    result: Dict[str, Any] = {
        "job_id": job_id,
        "status": "completed",
        "upload_id": upload_id,
        "model_name": "SEN2SRLite/NonReference_RGBN_x4",
        "input_shape": list(preprocessed.tensor.shape),
        "output_shape": list(mean_prediction.shape),
        "resolution": {
            "native_resolution_m": NATIVE_RESOLUTION_M,
            "sr_resolution_m": NATIVE_RESOLUTION_M / RGBN_SCALE_FACTOR,
            "scale_factor": RGBN_SCALE_FACTOR,
            "description": SR_PRODUCT_DESCRIPTION,
        },
        "bands": list(RGBN_BANDS),
        "crs": output_metadata.crs,
        "uncertainty": {
            "label": UNCERTAINTY_LABEL,
            "scalar_summary": uncertainty_result.scalar_summary,
            "scalar_summary_definition": uncertainty_result.scalar_summary_definition,
            "overall_distribution": dataclasses.asdict(uncertainty_result.overall_distribution),
            "n": uncertainty_result.n,
            "seed": uncertainty_result.seed,
            "transform_names": list(uncertainty_result.transform_names),
            "disclaimer": UNCERTAINTY_DISCLAIMER,
        },
        "self_consistency": {
            "downsample_rmse": consistency.downsample_consistency.overall.rmse,
            "ndvi_discrepancy_mean_abs": consistency.ndvi_comparison.mean_abs_discrepancy,
            "b08_b04_ratio_discrepancy_mean_abs": consistency.b08_b04_ratio_comparison.mean_abs_discrepancy,
            "note": "Compares the SR output against its own LR input only -- not a ground-truth accuracy check.",
        },
        "metadata": {
            "inference_seconds": round(inference_seconds, 4),
            "device": device,
            "output_geospatial": {
                "crs": output_metadata.crs,
                "transform": list(output_metadata.transform),
                "bounds": list(output_metadata.bounds),
                "width": output_metadata.width,
                "height": output_metadata.height,
            },
            "preprocessing_mask_coverage": preprocessed.mask.coverage(),
        },
        "scientific_caveats": list(SCIENTIFIC_CAVEATS),
        "artifacts": {
            "sr_geotiff": str(sr_mean_path),
            "uncertainty_geotiff": str(sr_std_path),
        },
        "created_at": _now_iso(),
    }

    record = SRJobRecord(
        job_id=job_id,
        status="completed",
        upload_id=upload_id,
        input_tensor_path=input_tensor_path,
        sr_mean_path=sr_mean_path,
        sr_std_path=sr_std_path,
        sr_mean_tensor_path=sr_mean_tensor_path,
        sr_std_tensor_path=sr_std_tensor_path,
        input_metadata=preprocessed.metadata,
        output_metadata=output_metadata,
        result=result,
        error=None,
        created_at=result["created_at"],
    )
    store.put_job(record)
    return record


# ---------------------------------------------------------------------------
# POST /analysis/ndvi
# ---------------------------------------------------------------------------

def run_ndvi_analysis_job(store: JobStore, job_id: str) -> AnalysisRecord:
    """Run the Phase 6 NDVI analysis against a completed SR job's already-
    saved tensors -- no model inference, no re-fetch."""
    job = store.get_job(job_id)

    lr_reflectance = torch.load(job.input_tensor_path, weights_only=True)
    sr_mean_prediction = torch.load(job.sr_mean_tensor_path, weights_only=True)
    sr_std_prediction = torch.load(job.sr_std_tensor_path, weights_only=True)

    report = run_ndvi_analysis(
        lr_reflectance,
        sr_mean_prediction,
        sr_std_prediction,
        band_names=RGBN_BANDS,
        scale_factor=RGBN_SCALE_FACTOR,
        native_resolution_m=NATIVE_RESOLUTION_M,
    )

    analysis_id = store.generate_id()

    # Single-band NDVI GeoTIFFs -- the frontend's NDVI view needs actual
    # rasters (native NDVI, SR-derived NDVI, their difference), not just the
    # scalar summary above. `report.native_ndvi`/`sr_ndvi`/
    # `ndvi_comparison.absolute_difference_map` already hold these arrays
    # (frame.analysis computed them; nothing is recomputed here). Written
    # into the same per-job directory as the SR artifacts, using the exact
    # geospatial metadata each grid was already derived with.
    job_dir = job.input_tensor_path.parent
    native_ndvi_metadata = dataclasses.replace(job.input_metadata, band_names=("NDVI",), nodata_value=float("nan"))
    sr_ndvi_metadata = dataclasses.replace(job.output_metadata, band_names=("NDVI",), nodata_value=float("nan"))

    native_ndvi_path = job_dir / f"ndvi_native_{analysis_id}.tif"
    write_geotiff(native_ndvi_path, report.native_ndvi.ndvi.numpy()[None], native_ndvi_metadata)

    sr_ndvi_path = job_dir / f"ndvi_sr_{analysis_id}.tif"
    write_geotiff(sr_ndvi_path, report.sr_ndvi.ndvi.numpy()[None], sr_ndvi_metadata)

    ndvi_diff_path = job_dir / f"ndvi_diff_{analysis_id}.tif"
    write_geotiff(ndvi_diff_path, report.ndvi_comparison.absolute_difference_map.numpy()[None], native_ndvi_metadata)

    result: Dict[str, Any] = {
        "analysis_id": analysis_id,
        "job_id": job_id,
        "status": "completed",
        "ndvi_formula": report.metadata["ndvi_formula"],
        "native_resolution_m": report.metadata["native_resolution_m"],
        "sr_resolution_m": report.metadata["sr_resolution_m"],
        "comparison": {
            "status": report.ndvi_comparison.comparison.status.value,
            "valid_pixel_count": report.ndvi_comparison.comparison.valid_pixel_count,
            "mean_abs_difference": report.ndvi_comparison.comparison.mean_abs_difference,
            "rmse": report.ndvi_comparison.comparison.rmse,
            "max_abs_difference": report.ndvi_comparison.comparison.max_abs_difference,
            "resampling_method": report.ndvi_comparison.comparison.resampling_method,
        },
        "uncertainty_weighted_summary": {
            "status": report.uncertainty_weighted_summary.status.value,
            "correlation_uncertainty_vs_abs_diff": report.uncertainty_weighted_summary.correlation_uncertainty_vs_abs_diff,
            "uncertainty_weighted_mean_abs_diff": report.uncertainty_weighted_summary.uncertainty_weighted_mean_abs_diff,
            "unweighted_mean_abs_diff": report.uncertainty_weighted_summary.unweighted_mean_abs_diff,
        },
        "scientific_caveats": list(report.scientific_caveats),
        "metadata": report.metadata,
        "artifacts": {
            "native_ndvi_geotiff": str(native_ndvi_path),
            "sr_ndvi_geotiff": str(sr_ndvi_path),
            "ndvi_diff_geotiff": str(ndvi_diff_path),
        },
        "created_at": _now_iso(),
    }

    record = AnalysisRecord(
        analysis_id=analysis_id,
        job_id=job_id,
        native_ndvi_path=native_ndvi_path,
        sr_ndvi_path=sr_ndvi_path,
        ndvi_diff_path=ndvi_diff_path,
        result=result,
        created_at=result["created_at"],
    )
    store.put_analysis(record)
    return record
