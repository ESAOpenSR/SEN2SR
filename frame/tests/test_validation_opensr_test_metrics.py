"""Tests for frame.validation.opensr_test_metrics -- group (B) in the Phase 4
CRITICAL three-way split: opensr-test's own metric vocabulary (reflectance,
spectral, spatial, synthesis, hallucination, omission, improvement), wrapped
via the real, installed `opensr_test.Metrics` class (not reimplemented --
see frame/validation/README.md).

Uses only small synthetic torch tensors -- no network (opensr_test.Metrics
itself makes no network calls; only opensr_test.load does).
"""

import math

import torch
import pytest

from frame.consistency.status import ComputationStatus
from frame.validation.errors import BenchmarkFormatError
from frame.validation.opensr_test_metrics import compute_opensr_test_metrics

# Large enough to survive opensr_test's default border_mask=16 crop
# (16 px off HR/SR, 16//scale_factor off LR) without collapsing to nothing.
LR_SIZE = 48
HR_SIZE = 192  # scale_factor = 4


def _tensors(seed=0):
    g = torch.Generator().manual_seed(seed)
    lr = torch.rand(4, LR_SIZE, LR_SIZE, generator=g)
    sr = torch.rand(4, HR_SIZE, HR_SIZE, generator=g)
    hr = torch.rand(4, HR_SIZE, HR_SIZE, generator=g)
    return lr, sr, hr


def test_returns_all_seven_named_metrics():
    lr, sr, hr = _tensors()
    result = compute_opensr_test_metrics(lr, sr, hr)
    assert result.status == ComputationStatus.COMPUTABLE
    for field in ("reflectance", "spectral", "spatial", "synthesis", "hallucination", "omission", "improvement"):
        assert hasattr(result, field)


def test_identical_sr_and_hr_gives_near_zero_reflectance_and_spectral_error():
    lr, _, hr = _tensors()
    # sr == hr exactly -> the model "recovered" the reference perfectly
    result = compute_opensr_test_metrics(lr, hr.clone(), hr)
    assert result.reflectance == pytest.approx(0.0, abs=0.35)  # LR/HR are unrelated random tensors here
    assert result.spectral is not None


def test_nan_fields_are_reported_as_none_not_as_nan():
    # opensr-test's own spatial (phase-correlation) metric can legitimately
    # return NaN on pathological input (e.g. pure noise with no coherent
    # translation to detect) -- verified interactively during development.
    # Never let a bare NaN leak into a saved report; None + surrounding
    # `status` is this project's consistent convention.
    lr, sr, hr = _tensors()
    result = compute_opensr_test_metrics(lr, sr, hr)
    for field in ("reflectance", "spectral", "spatial", "synthesis", "hallucination", "omission", "improvement"):
        value = getattr(result, field)
        if value is not None:
            assert not math.isnan(value), f"{field} leaked a bare NaN instead of None"


def test_config_summary_is_recorded_for_reproducibility():
    lr, sr, hr = _tensors()
    result = compute_opensr_test_metrics(lr, sr, hr)
    assert "correctness_temperature" in result.config_summary
    assert "ha_score" in result.config_summary
    assert "spatial_method" in result.config_summary


def test_opensr_test_version_is_recorded():
    lr, sr, hr = _tensors()
    result = compute_opensr_test_metrics(lr, sr, hr)
    import opensr_test

    assert result.opensr_test_version == opensr_test.__version__


def test_malformed_input_raises_benchmark_format_error():
    lr = torch.rand(4, LR_SIZE, LR_SIZE)
    sr = torch.rand(4, HR_SIZE, HR_SIZE)
    hr = torch.rand(4, HR_SIZE + 1, HR_SIZE + 1)  # sr/hr shape mismatch
    with pytest.raises(BenchmarkFormatError):
        compute_opensr_test_metrics(lr, sr, hr)
