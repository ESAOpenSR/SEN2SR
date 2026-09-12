"""Tests for frame.api.errors -- the API-level exception taxonomy and its
mapping onto HTTP status codes. Pure logic, no FastAPI app needed.
"""

import pytest

from frame.api.errors import (
    APIError,
    JobNotFoundError,
    UnsupportedFileError,
    ValidationFailedError,
    status_code_for,
)
from frame.consistency.errors import ShapeMismatchError as ConsistencyShapeMismatchError
from frame.geospatial.errors import MissingCRSError
from frame.preprocessing.errors import UnsupportedBandsError


def test_job_not_found_maps_to_404():
    assert status_code_for(JobNotFoundError("no such job")) == 404


def test_unsupported_file_maps_to_400():
    assert status_code_for(UnsupportedFileError("bad file")) == 400


def test_validation_failed_maps_to_422():
    assert status_code_for(ValidationFailedError("bad request")) == 422


def test_frame_preprocessing_error_maps_to_400():
    assert status_code_for(UnsupportedBandsError("missing band")) == 400


def test_frame_geospatial_error_maps_to_400():
    assert status_code_for(MissingCRSError("no crs")) == 400


def test_frame_consistency_error_maps_to_400():
    assert status_code_for(ConsistencyShapeMismatchError("bad shape")) == 400


def test_unrecognized_exception_maps_to_500():
    assert status_code_for(RuntimeError("something broke")) == 500


def test_api_error_carries_a_machine_readable_code():
    err = JobNotFoundError("no such job", job_id="abc123")
    assert err.code == "job_not_found"
    assert "abc123" in str(err)


def test_every_api_error_subclass_is_an_api_error():
    assert issubclass(JobNotFoundError, APIError)
    assert issubclass(UnsupportedFileError, APIError)
    assert issubclass(ValidationFailedError, APIError)
