"""Tests for frame.analysis.uncertainty_overlay -- aggregating Phase 5's
per-band uncertainty to a single map and relating it to NDVI disagreement
(Phase 6). No confidence threshold is invented anywhere here.
"""

import numpy as np
import torch
import pytest

from frame.consistency import ComputationStatus
from frame.analysis.uncertainty_overlay import (
    UncertaintyWeightedNDVISummary,
    aggregate_uncertainty_overall,
    compute_uncertainty_weighted_ndvi_summary,
)


# ---------------------------------------------------------------------------
# aggregate_uncertainty_overall -- 8. uncertainty shape alignment
# ---------------------------------------------------------------------------

def test_aggregate_reduces_band_axis_only():
    std_prediction = torch.rand(4, 32, 32)
    overall = aggregate_uncertainty_overall(std_prediction)
    assert overall.shape == (32, 32)


def test_aggregate_matches_hand_computed_mean_over_bands():
    std_prediction = torch.stack([torch.full((4, 4), v) for v in (0.1, 0.2, 0.3, 0.4)])
    overall = aggregate_uncertainty_overall(std_prediction)
    assert torch.allclose(overall, torch.full((4, 4), 0.25), atol=1e-6)


def test_aggregate_rejects_non_3d_input():
    with pytest.raises(Exception):
        aggregate_uncertainty_overall(torch.rand(32, 32))


# ---------------------------------------------------------------------------
# compute_uncertainty_weighted_ndvi_summary
# ---------------------------------------------------------------------------

def test_summary_reports_correlation_and_weighted_and_unweighted_means():
    torch.manual_seed(0)
    uncertainty = torch.rand(8, 8)
    abs_diff = uncertainty * 2.0  # perfectly correlated by construction
    mask = np.ones((8, 8), dtype=bool)
    summary = compute_uncertainty_weighted_ndvi_summary(uncertainty, abs_diff, mask)
    assert isinstance(summary, UncertaintyWeightedNDVISummary)
    assert summary.status == ComputationStatus.COMPUTABLE
    assert summary.correlation_uncertainty_vs_abs_diff == pytest.approx(1.0, abs=1e-3)
    assert summary.unweighted_mean_abs_diff == pytest.approx(float(abs_diff.mean()), abs=1e-5)


def test_summary_handles_uncorrelated_case():
    torch.manual_seed(1)
    uncertainty = torch.rand(8, 8)
    abs_diff = torch.full((8, 8), 0.2)  # constant -> zero variance -> undefined correlation
    mask = np.ones((8, 8), dtype=bool)
    summary = compute_uncertainty_weighted_ndvi_summary(uncertainty, abs_diff, mask)
    assert summary.correlation_uncertainty_vs_abs_diff is None  # not computable, not fabricated as 0


def test_uncertainty_weighted_mean_is_none_when_uncertainty_sums_to_zero():
    uncertainty = torch.zeros(8, 8)
    abs_diff = torch.rand(8, 8)
    mask = np.ones((8, 8), dtype=bool)
    summary = compute_uncertainty_weighted_ndvi_summary(uncertainty, abs_diff, mask)
    assert summary.uncertainty_weighted_mean_abs_diff is None


def test_masked_pixels_excluded_from_summary():
    uncertainty = torch.ones(8, 8)
    abs_diff = torch.zeros(8, 8)
    abs_diff[0, 0] = 99.0  # huge outlier, masked out
    mask = np.ones((8, 8), dtype=bool)
    mask[0, 0] = False
    summary = compute_uncertainty_weighted_ndvi_summary(uncertainty, abs_diff, mask)
    assert summary.unweighted_mean_abs_diff == pytest.approx(0.0, abs=1e-6)


def test_all_invalid_mask_is_not_computable():
    uncertainty = torch.rand(8, 8)
    abs_diff = torch.rand(8, 8)
    mask = np.zeros((8, 8), dtype=bool)
    summary = compute_uncertainty_weighted_ndvi_summary(uncertainty, abs_diff, mask)
    assert summary.status == ComputationStatus.NOT_COMPUTABLE
    assert summary.correlation_uncertainty_vs_abs_diff is None
    assert summary.uncertainty_weighted_mean_abs_diff is None
    assert summary.unweighted_mean_abs_diff is None


def test_summary_does_not_carry_a_threshold_field():
    import dataclasses

    field_names = {f.name.lower() for f in dataclasses.fields(UncertaintyWeightedNDVISummary)}
    for forbidden in ("threshold", "is_high", "is_reliable", "confidence"):
        assert forbidden not in field_names
