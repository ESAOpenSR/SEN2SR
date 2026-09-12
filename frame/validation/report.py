"""Top-level entry point: `run_validation_sample` -- Phase 4's reference-based
validation orchestrator.

Produces a `ValidationReport` that keeps three metric groups explicitly
separate, per this phase's CRITICAL requirement -- they are never averaged,
weighted, or merged into one invented score:

  (A) `standard_reference_metrics` -- PSNR/SSIM/RMSE/SAM/ERGAS, computed for
      BOTH the bicubic baseline and the real SEN2SR output against the
      genuine HR reference (`frame.validation.reference_metrics`), plus a
      per-metric bicubic-vs-SEN2SR comparison.
  (B) `opensr_test_metrics` -- opensr-test's own reflectance/spectral/
      spatial/synthesis/hallucination/omission/improvement vocabulary,
      computed via the real `opensr_test.Metrics` class
      (`frame.validation.opensr_test_metrics`).
  (C) `phase3_self_consistency` -- this project's own Phase 3 no-reference
      diagnostics (`frame.consistency.run_consistency_diagnostics`), which
      need no HR at all and are reused here unchanged, not reimplemented.

Scientific framing (docs/FRAME_TECHNICAL_SPEC.md Section 1.4/11, restated
verbatim on every report via `SCIENTIFIC_FRAMING`): this benchmark evaluates
reconstruction against an independently sourced higher-resolution reference
dataset. It does not, and cannot, establish that the SR output equals a
native 2.5 m Sentinel-2 observation, because no such native Sentinel-2
measurement exists anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from frame.consistency import ConsistencyDiagnostics, run_consistency_diagnostics
from frame.preprocessing.masks import ValidityMask
from frame.validation.bicubic import bicubic_upsample
from frame.validation.opensr_test_adapter import OpenSRTestSample
from frame.validation.opensr_test_metrics import OpenSRTestMetrics, compute_opensr_test_metrics
from frame.validation.reference_metrics import ReferenceMetrics, compute_reference_metrics

# Which reference metrics improve by going up vs. down -- used only to
# label each comparison, never to compute a combined/weighted score.
HIGHER_IS_BETTER: Dict[str, bool] = {
    "psnr_db": True,
    "ssim": True,
    "rmse": False,
    "sam_degrees": False,
    "ergas": False,
}

SCIENTIFIC_FRAMING = (
    "This benchmark evaluates reconstruction against an independently sourced "
    "higher-resolution reference dataset (opensr-test, itself built from NAIP/"
    "SPOT/Spanish IGN aerial or satellite imagery -- a different sensor, "
    "acquisition date, and platform than Sentinel-2). It does NOT establish "
    "that the SR output equals a native 2.5m Sentinel-2 observation, because "
    "no such native Sentinel-2 measurement exists: Sentinel-2's finest native "
    "band resolution is 10 m. Every metric below characterizes agreement with "
    "this specific external reference, not a proof of ground-truth accuracy "
    "at 2.5 m. Cross-sensor confounds (acquisition-date mismatch, atmospheric/"
    "illumination differences, residual co-registration error) apply to every "
    "number in group (A) and (B) below; group (C) needs no external reference "
    "at all and carries the self-consistency limitations documented in "
    "frame/consistency/README.md instead."
)


@dataclass(frozen=True)
class MetricComparison:
    bicubic_value: Optional[float]
    sen2sr_value: Optional[float]
    absolute_change: Optional[float]  # sen2sr_value - bicubic_value
    relative_change: Optional[float]  # absolute_change / abs(bicubic_value), when defined
    higher_is_better: bool


@dataclass(frozen=True)
class BicubicVsSR:
    bicubic: ReferenceMetrics
    sen2sr: ReferenceMetrics
    comparisons: Dict[str, MetricComparison]


@dataclass(frozen=True)
class ValidationReport:
    subset: str
    sample_index: int
    roi_id: Optional[str]
    standard_reference_metrics: BicubicVsSR
    opensr_test_metrics: OpenSRTestMetrics
    phase3_self_consistency: ConsistencyDiagnostics
    scientific_framing: str
    parameters: Dict[str, Any]


def _compare(bicubic: ReferenceMetrics, sen2sr: ReferenceMetrics) -> Dict[str, MetricComparison]:
    comparisons: Dict[str, MetricComparison] = {}
    for field, higher_is_better in HIGHER_IS_BETTER.items():
        b = getattr(bicubic, field)
        s = getattr(sen2sr, field)
        if b is None or s is None or not np.isfinite(b) or not np.isfinite(s):
            comparisons[field] = MetricComparison(b, s, None, None, higher_is_better)
            continue
        absolute_change = s - b
        relative_change = absolute_change / abs(b) if b != 0 else None
        comparisons[field] = MetricComparison(b, s, absolute_change, relative_change, higher_is_better)
    return comparisons


def run_validation_sample(
    sample: OpenSRTestSample,
    sr: torch.Tensor,
    *,
    lr_mask: Optional[np.ndarray] = None,
    hr_mask: Optional[np.ndarray] = None,
    nodata_value: float = 0.0,
) -> ValidationReport:
    """Run every Phase 4 validation group against one benchmark sample.

    Args:
        sample: An `OpenSRTestSample` (see `frame.validation.opensr_test_adapter`).
        sr: The real, already-computed SEN2SR output for `sample.lr_reflectance`,
            shape matching `sample.hr_reflectance`.
        lr_mask, hr_mask: Optional boolean validity masks (LR grid / HR grid
            respectively). If not given, derived from each grid's own
            nodata pixels (`frame.preprocessing.ValidityMask.from_nodata`) --
            opensr-test's own datasets are curated to minimize misalignment,
            so this is a defensive safety net (catching reprojection-border
            fill values), not the primary quality-control mechanism.
        nodata_value: Value treated as nodata when deriving default masks.
    """
    lr = sample.lr_reflectance
    hr = sample.hr_reflectance

    if lr_mask is None:
        lr_mask = ValidityMask.from_nodata(lr.numpy(), nodata_value=nodata_value).array
    if hr_mask is None:
        hr_mask = ValidityMask.from_nodata(hr.numpy(), nodata_value=nodata_value).array

    bicubic = bicubic_upsample(lr, sample.scale_factor)

    bicubic_ref = compute_reference_metrics(
        bicubic, hr, hr_mask, band_names=sample.hr_metadata.band_names, scale_factor=sample.scale_factor
    )
    sen2sr_ref = compute_reference_metrics(
        sr, hr, hr_mask, band_names=sample.hr_metadata.band_names, scale_factor=sample.scale_factor
    )
    standard_reference_metrics = BicubicVsSR(
        bicubic=bicubic_ref, sen2sr=sen2sr_ref, comparisons=_compare(bicubic_ref, sen2sr_ref)
    )

    opensr_test_metrics = compute_opensr_test_metrics(lr, sr, hr)

    phase3_self_consistency = run_consistency_diagnostics(
        lr=lr, sr=sr, mask=lr_mask, band_names=sample.lr_metadata.band_names, scale_factor=sample.scale_factor
    )

    parameters: Dict[str, Any] = {
        "subset": sample.subset,
        "sample_index": sample.sample_index,
        "roi_id": sample.roi_id,
        "hr_variant": sample.hr_variant,
        "scale_factor": sample.scale_factor,
        "dataset_version": sample.dataset_version,
        "l2a_band_order_source": sample.l2a_band_order_source,
        "band_names": tuple(sample.hr_metadata.band_names),
        "nodata_value": nodata_value,
    }

    return ValidationReport(
        subset=sample.subset,
        sample_index=sample.sample_index,
        roi_id=sample.roi_id,
        standard_reference_metrics=standard_reference_metrics,
        opensr_test_metrics=opensr_test_metrics,
        phase3_self_consistency=phase3_self_consistency,
        scientific_framing=SCIENTIFIC_FRAMING,
        parameters=parameters,
    )
