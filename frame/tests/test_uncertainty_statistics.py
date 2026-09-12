"""Tests for frame.uncertainty.statistics -- the streaming (Welford) mean/
variance accumulator, distribution-summary statistics, and the visualization-
only normalization helper (Phase 5).
"""

import numpy as np
import torch
import pytest

from frame.uncertainty.errors import ShapeMismatchError
from frame.uncertainty.statistics import (
    UncertaintyDistributionStats,
    WelfordAccumulator,
    compute_distribution_stats,
    normalize_for_visualization,
)


# ---------------------------------------------------------------------------
# WelfordAccumulator -- hand-computable toy examples
# ---------------------------------------------------------------------------

def test_single_update_gives_that_value_as_mean_and_zero_variance():
    acc = WelfordAccumulator()
    x = torch.tensor([1.0, 2.0, 3.0])
    acc.update(x)
    assert torch.equal(acc.mean, x)
    assert torch.allclose(acc.variance, torch.zeros_like(x))
    assert torch.allclose(acc.std, torch.zeros_like(x))


def test_mean_matches_hand_computed_example():
    # three scalar "images" of shape (1,): 2, 4, 6 -> mean = 4
    acc = WelfordAccumulator()
    for v in (2.0, 4.0, 6.0):
        acc.update(torch.tensor([v]))
    assert torch.allclose(acc.mean, torch.tensor([4.0]))


def test_population_variance_matches_hand_computed_example():
    # values 2, 4, 6 -> mean 4 -> squared deviations 4, 0, 4 -> population variance = 8/3
    acc = WelfordAccumulator()
    for v in (2.0, 4.0, 6.0):
        acc.update(torch.tensor([v]))
    assert torch.allclose(acc.variance, torch.tensor([8.0 / 3.0]), atol=1e-6)
    assert torch.allclose(acc.std, torch.tensor([(8.0 / 3.0) ** 0.5]), atol=1e-6)


def test_variance_is_zero_for_identical_repeated_updates():
    acc = WelfordAccumulator()
    x = torch.full((4, 8, 8), 0.5)
    for _ in range(6):
        acc.update(x.clone())
    assert torch.allclose(acc.variance, torch.zeros_like(x), atol=1e-8)
    assert torch.allclose(acc.std, torch.zeros_like(x), atol=1e-8)


def test_variance_is_positive_when_updates_differ():
    acc = WelfordAccumulator()
    acc.update(torch.zeros(4, 8, 8))
    acc.update(torch.ones(4, 8, 8))
    assert torch.all(acc.variance > 0)


def test_accessing_mean_before_any_update_raises():
    acc = WelfordAccumulator()
    with pytest.raises(ShapeMismatchError):
        _ = acc.mean


def test_count_tracks_number_of_updates():
    acc = WelfordAccumulator()
    assert acc.count == 0
    acc.update(torch.zeros(2, 2))
    acc.update(torch.ones(2, 2))
    assert acc.count == 2


def test_update_rejects_a_shape_that_does_not_match_the_running_accumulator():
    acc = WelfordAccumulator()
    acc.update(torch.zeros(4, 8, 8))
    with pytest.raises(ShapeMismatchError):
        acc.update(torch.zeros(4, 4, 4))


# ---------------------------------------------------------------------------
# compute_distribution_stats
# ---------------------------------------------------------------------------

def test_distribution_stats_on_a_hand_computable_map():
    # a 2x2 map with known values -> mean/median/std/min/max all hand-checkable
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stats = compute_distribution_stats(values)
    assert isinstance(stats, UncertaintyDistributionStats)
    assert stats.mean == pytest.approx(2.5)
    assert stats.median == pytest.approx(2.5)
    assert stats.min == pytest.approx(1.0)
    assert stats.max == pytest.approx(4.0)
    assert stats.std == pytest.approx(float(np.std([1, 2, 3, 4])), abs=1e-6)


def test_distribution_stats_reports_p90_and_p95():
    values = torch.arange(1, 101).float()  # 1..100
    stats = compute_distribution_stats(values)
    assert stats.p90 == pytest.approx(90.1, abs=1.0)
    assert stats.p95 == pytest.approx(95.05, abs=1.0)


def test_distribution_stats_does_not_carry_a_pass_fail_threshold_field():
    import dataclasses

    field_names = {f.name.lower() for f in dataclasses.fields(UncertaintyDistributionStats)}
    for forbidden in ("threshold", "is_high", "is_reliable", "passed", "good", "bad"):
        assert forbidden not in field_names


# ---------------------------------------------------------------------------
# normalize_for_visualization -- display-only, never touches the real values
# ---------------------------------------------------------------------------

def test_normalize_for_visualization_maps_to_zero_one_range():
    values = torch.tensor([[0.0, 5.0], [10.0, 100.0]])
    normalized = normalize_for_visualization(values, low_percentile=0, high_percentile=100)
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0 + 1e-6


def test_normalize_for_visualization_does_not_mutate_the_input():
    values = torch.tensor([[0.0, 5.0], [10.0, 100.0]])
    original = values.clone()
    normalize_for_visualization(values)
    assert torch.equal(values, original)


def test_normalize_for_visualization_constant_map_does_not_divide_by_zero():
    values = torch.full((4, 4), 0.3)
    normalized = normalize_for_visualization(values)
    assert torch.isfinite(normalized).all()
