"""FRAME downstream analysis layer -- Phase 6's first lightweight demonstration.

Computes NDVI from the existing 4-band RGBN pipeline at both the native
10 m grid and the SR 2.5 m pixel grid (Phase 5's mean SR prediction),
compares them on a common grid (never a raw 2.5 m-vs-10 m comparison --
the SR-grid NDVI is always reduced first, via
`frame.consistency.downsample_to_lr_grid`, reused unchanged), and relates
that comparison to Phase 5's per-pixel model-stability uncertainty.

**This is not a claim that 2.5 m NDVI is ground truth.** It demonstrates
that FRAME can provide a finer-grained vegetation-index visualization while
explicitly exposing model-stability uncertainty alongside it -- see
README.md's four scientific caveats
(`frame.analysis.report.SCIENTIFIC_CAVEATS`), carried on every
`NDVIAnalysisReport`.

This package imports nothing from `sen2sr` and calls no model itself, and
does not call `frame.uncertainty`'s ensemble loop either -- it consumes
Phase 5's already-computed `mean_prediction`/`std_prediction` directly, per
this phase's explicit instruction not to rerun unnecessary model inference.
See `experiments/analysis/run_experiment.py` for the real integration.

Public API:
    run_ndvi_analysis, NDVIAnalysisReport, SCIENTIFIC_CAVEATS, NDVI_FORMULA  (report.py)
    compute_ndvi_from_stack, NDVIResult                                     (indices.py)
    compare_ndvi_on_common_grid, downsample_ndvi_to_native_grid,
        NDVIComparison, NDVIComparisonResult                                (comparison.py)
    aggregate_uncertainty_overall, compute_uncertainty_weighted_ndvi_summary,
        UncertaintyWeightedNDVISummary                                      (uncertainty_overlay.py)
    Exceptions: AnalysisError and its subclasses (see .errors)
"""

from frame.analysis.comparison import (
    NDVIComparison,
    NDVIComparisonResult,
    compare_ndvi_on_common_grid,
    downsample_ndvi_to_native_grid,
)
from frame.analysis.errors import AnalysisError, MissingBandError, ShapeMismatchError
from frame.analysis.indices import NDVIResult, compute_ndvi_from_stack
from frame.analysis.report import NDVI_FORMULA, SCIENTIFIC_CAVEATS, NDVIAnalysisReport, run_ndvi_analysis
from frame.analysis.uncertainty_overlay import (
    UncertaintyWeightedNDVISummary,
    aggregate_uncertainty_overall,
    compute_uncertainty_weighted_ndvi_summary,
)

__all__ = [
    "run_ndvi_analysis",
    "NDVIAnalysisReport",
    "SCIENTIFIC_CAVEATS",
    "NDVI_FORMULA",
    "compute_ndvi_from_stack",
    "NDVIResult",
    "compare_ndvi_on_common_grid",
    "downsample_ndvi_to_native_grid",
    "NDVIComparison",
    "NDVIComparisonResult",
    "aggregate_uncertainty_overall",
    "compute_uncertainty_weighted_ndvi_summary",
    "UncertaintyWeightedNDVISummary",
    "AnalysisError",
    "MissingBandError",
    "ShapeMismatchError",
]
