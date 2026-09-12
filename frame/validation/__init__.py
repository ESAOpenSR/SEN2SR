"""FRAME reference-based validation layer.

Runs FRAME's proven `SEN2SRLite/NonReference_RGBN_x4` path (unmodified)
against the `opensr-test` benchmark's real LR/HR pairs, producing three
explicitly-separate metric groups (never merged into one invented score):
(A) standard reference metrics (PSNR/SSIM/RMSE/SAM/ERGAS) for both a
bicubic baseline and the real SR output, (B) opensr-test's own metric
vocabulary, and (C) this project's own Phase 3 self-consistency diagnostics.

See docs/FRAME_TECHNICAL_SPEC.md Section 11.B for the design this
implements, README.md in this package for the concrete contract, and
experiments/validation/ for the reference integration against a real
benchmark subset and a real model run.

Public API:
    run_validation_sample, ValidationReport, BicubicVsSR, MetricComparison,
        SCIENTIFIC_FRAMING, HIGHER_IS_BETTER                  (report.py)
    load_subset, extract_sample, OpenSRTestSample,
        SUPPORTED_SUBSETS, L2A_BAND_ORDER, RGBN_BAND_NAMES     (opensr_test_adapter.py)
    bicubic_upsample                                            (bicubic.py)
    compute_reference_metrics, ReferenceMetrics                 (reference_metrics.py)
    compute_opensr_test_metrics, OpenSRTestMetrics              (opensr_test_metrics.py)
    Exceptions: ValidationError and its subclasses (see .errors)
"""

from frame.validation.bicubic import bicubic_upsample
from frame.validation.errors import (
    BenchmarkFormatError,
    SampleIndexError,
    ShapeMismatchError,
    UnsupportedSubsetError,
    ValidationError,
)
from frame.validation.opensr_test_adapter import (
    L2A_BAND_ORDER,
    RGBN_BAND_NAMES,
    SUPPORTED_SUBSETS,
    OpenSRTestSample,
    extract_sample,
    load_subset,
)
from frame.validation.opensr_test_metrics import OpenSRTestMetrics, compute_opensr_test_metrics
from frame.validation.reference_metrics import ReferenceMetrics, compute_reference_metrics
from frame.validation.report import (
    HIGHER_IS_BETTER,
    SCIENTIFIC_FRAMING,
    BicubicVsSR,
    MetricComparison,
    ValidationReport,
    run_validation_sample,
)

__all__ = [
    "run_validation_sample",
    "ValidationReport",
    "BicubicVsSR",
    "MetricComparison",
    "SCIENTIFIC_FRAMING",
    "HIGHER_IS_BETTER",
    "load_subset",
    "extract_sample",
    "OpenSRTestSample",
    "SUPPORTED_SUBSETS",
    "L2A_BAND_ORDER",
    "RGBN_BAND_NAMES",
    "bicubic_upsample",
    "compute_reference_metrics",
    "ReferenceMetrics",
    "compute_opensr_test_metrics",
    "OpenSRTestMetrics",
    "ValidationError",
    "UnsupportedSubsetError",
    "SampleIndexError",
    "ShapeMismatchError",
    "BenchmarkFormatError",
]
