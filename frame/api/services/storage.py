"""In-memory upload/job/analysis registry for the FRAME API (Phase 7).

Deliberately a plain in-process dict, not a database or a distributed job
queue -- this is an SIH prototype/demo backend, and the task explicitly
asks not to invent Celery/Redis/queues unless absolutely necessary. State
is lost on server restart; generated artifacts on disk (under
`frame.api.config.WORKSPACE_DIR`) persist independently of this registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from frame.api.errors import APIError, JobNotFoundError, UploadNotFoundError
from frame.preprocessing.metadata import RasterMetadata


class AnalysisNotFoundError(APIError):
    code = "analysis_not_found"
    status_code = 404

    def __init__(self, message: str, *, analysis_id: str = ""):
        super().__init__(f"{message} (analysis_id={analysis_id!r})" if analysis_id else message)
        self.analysis_id = analysis_id


@dataclass
class UploadRecord:
    upload_id: str
    filename: str
    file_path: Path
    band_names: Tuple[str, ...]
    width: int
    height: int
    crs: Optional[str]
    resolution_m: Optional[float]
    input_scale: str
    created_at: str


@dataclass
class SRJobRecord:
    job_id: str
    status: str  # "completed" | "failed"
    upload_id: str
    input_tensor_path: Path
    sr_mean_path: Path  # GeoTIFF
    sr_std_path: Path  # GeoTIFF
    sr_mean_tensor_path: Path  # .pt, reused by /analysis/ndvi
    sr_std_tensor_path: Path  # .pt, reused by /analysis/ndvi
    input_metadata: RasterMetadata  # native 10 m grid -- reused by /analysis/ndvi to export NDVI GeoTIFFs
    output_metadata: RasterMetadata  # SR 2.5 m pixel grid -- ditto
    result: Dict[str, Any]
    error: Optional[str]
    created_at: str


@dataclass
class AnalysisRecord:
    analysis_id: str
    job_id: str
    native_ndvi_path: Path  # single-band GeoTIFF, native 10 m grid
    sr_ndvi_path: Path  # single-band GeoTIFF, SR 2.5 m pixel grid
    ndvi_diff_path: Path  # single-band GeoTIFF, native 10 m grid
    result: Dict[str, Any]
    created_at: str


class JobStore:
    """A small in-memory registry. One instance per running API process."""

    def __init__(self) -> None:
        self._uploads: Dict[str, UploadRecord] = {}
        self._jobs: Dict[str, SRJobRecord] = {}
        self._analyses: Dict[str, AnalysisRecord] = {}

    @staticmethod
    def generate_id() -> str:
        return uuid.uuid4().hex

    def put_upload(self, record: UploadRecord) -> None:
        self._uploads[record.upload_id] = record

    def get_upload(self, upload_id: str) -> UploadRecord:
        try:
            return self._uploads[upload_id]
        except KeyError:
            raise UploadNotFoundError("No such upload", upload_id=upload_id) from None

    def put_job(self, record: SRJobRecord) -> None:
        self._jobs[record.job_id] = record

    def get_job(self, job_id: str) -> SRJobRecord:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise JobNotFoundError("No such SR job", job_id=job_id) from None

    def put_analysis(self, record: AnalysisRecord) -> None:
        self._analyses[record.analysis_id] = record

    def get_analysis(self, analysis_id: str) -> AnalysisRecord:
        try:
            return self._analyses[analysis_id]
        except KeyError:
            raise AnalysisNotFoundError("No such analysis", analysis_id=analysis_id) from None
