"""Wraps `opensr_test.Metrics` -- group "(B) opensr-test metrics" in the
Phase 4 CRITICAL three-way split (docs/FRAME_TECHNICAL_SPEC.md Section 9.2's
distinction, extended here to reference-based validation).

This module calls the real, installed `opensr-test` package's own `Metrics`
class directly -- reflectance/spectral/spatial/synthesis/hallucination/
omission/improvement are computed exactly as upstream defines them (see
`frame/validation/README.md` for the exact formulas, read from
`opensr_test/main.py`, `distance.py`, and `docs/Metrics/correctness.md`),
using upstream's own default `Config()` (border_mask=16, correctness
temperature 0.25, ha/om/im scores 0.05, spatial_method "pcc", etc.) -- these
are the benchmark's own published calibration constants, not values FRAME
invents. The exact config used is recorded on every result for
reproducibility.

Deliberately NOT reimplemented from these formulas' math -- this module's
only job is calling the real class and giving its output a stable,
documented shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from frame.consistency.status import ComputationStatus
from frame.validation.errors import BenchmarkFormatError


@dataclass(frozen=True)
class OpenSRTestMetrics:
    status: ComputationStatus
    reflectance: Optional[float]
    spectral: Optional[float]
    spatial: Optional[float]
    synthesis: Optional[float]
    hallucination: Optional[float]
    omission: Optional[float]
    improvement: Optional[float]
    opensr_test_version: str
    config_summary: Dict[str, Any]


def _clean(value) -> Optional[float]:
    """None (never a bare NaN) if the value is missing or not-a-number --
    consistent with this project's rule (frame.consistency,
    frame.validation.reference_metrics) that an uncomputed quantity is
    reported as None, never as a silent NaN or a fabricated 0.0."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


def compute_opensr_test_metrics(
    lr: torch.Tensor, sr: torch.Tensor, hr: torch.Tensor
) -> OpenSRTestMetrics:
    """Run opensr-test's own `Metrics().compute(lr, sr, hr)` and wrap the result.

    Args:
        lr: LR input tensor, shape (C, H, W).
        sr: SR (or bicubic-baseline) tensor, shape (C, H*scale, W*scale).
        hr: Real HR reference tensor, same shape as ``sr``.

    Raises:
        BenchmarkFormatError: if `opensr_test.Metrics` itself rejects the
            input (e.g. an inconsistent scale factor) -- the original
            upstream exception is chained via `from`.
    """
    import opensr_test

    metrics = opensr_test.Metrics()
    try:
        result = metrics.compute(lr=lr, sr=sr.detach(), hr=hr)
    except Exception as exc:  # noqa: BLE001 -- re-raised with our own taxonomy, original preserved via `from`
        raise BenchmarkFormatError(f"opensr_test.Metrics().compute() rejected the input: {exc}") from exc

    return OpenSRTestMetrics(
        status=ComputationStatus.COMPUTABLE,
        reflectance=_clean(result.get("reflectance")),
        spectral=_clean(result.get("spectral")),
        spatial=_clean(result.get("spatial")),
        synthesis=_clean(result.get("synthesis")),
        hallucination=_clean(result.get("ha_metric")),
        omission=_clean(result.get("om_metric")),
        improvement=_clean(result.get("im_metric")),
        opensr_test_version=opensr_test.__version__,
        config_summary=metrics.params.model_dump(),
    )
