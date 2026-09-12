"""Tests for frame.uncertainty.report -- the top-level `run_stochastic_uncertainty`
orchestrator (Phase 5's desired core API).
"""

import torch
import torch.nn.functional as F
import pytest

from frame.uncertainty.errors import InvalidEnsembleConfigError, ShapeMismatchError
from frame.uncertainty.report import UncertaintyResult, run_stochastic_uncertainty
from frame.uncertainty.statistics import normalize_for_visualization
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS, IDENTITY

SCALE = 4
BANDS = ("B04", "B03", "B02", "B08")


def _equivariant_model(x_batched):
    return F.interpolate(x_batched, scale_factor=SCALE, mode="bicubic", antialias=True)


def _biased_model(x_batched):
    base = F.interpolate(x_batched, scale_factor=SCALE, mode="bicubic", antialias=True)
    bias = x_batched[0, 0, 0, 0].item() * 0.5
    return base + bias


def _input(seed=0, h=8, w=8):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(4, h, w, generator=g) * 0.5 + 0.1


def test_returns_the_desired_core_api_shape():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, transforms=DEFAULT_TRANSFORMS, seed=42, band_names=BANDS)
    assert isinstance(result, UncertaintyResult)
    assert result.mean_prediction.shape == (4, 32, 32)
    assert result.std_prediction.shape == (4, 32, 32)
    assert result.variance_prediction.shape == (4, 32, 32)
    assert isinstance(result.scalar_summary, float)
    assert isinstance(result.metadata, dict)


def test_defaults_to_the_six_geometric_transforms_when_none_given():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    assert result.n == 6
    assert result.transform_names == ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


def test_scalar_summary_equals_overall_distribution_mean_and_is_documented():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    assert result.scalar_summary == pytest.approx(result.overall_distribution.mean)
    assert "mean" in result.scalar_summary_definition.lower()
    assert "per-pixel" in result.scalar_summary_definition.lower()


def test_per_band_distribution_has_one_entry_per_band_name():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    assert set(result.per_band_distribution.keys()) == set(BANDS)


def test_overall_distribution_reports_the_required_summary_statistics():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    d = result.overall_distribution
    for field in ("mean", "median", "std", "p90", "p95", "max", "min"):
        assert hasattr(d, field)


def test_zero_uncertainty_for_a_perfectly_equivariant_model():
    x = _input()
    result = run_stochastic_uncertainty(_equivariant_model, x, seed=42, band_names=BANDS)
    assert result.scalar_summary == pytest.approx(0.0, abs=1e-4)
    assert result.overall_distribution.max == pytest.approx(0.0, abs=1e-3)


def test_per_transform_disagreement_is_recorded_for_every_transform():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    assert len(result.per_transform_disagreement) == 6
    names = {d.transform_name for d in result.per_transform_disagreement}
    assert names == {"identity", "hflip", "vflip", "rot90", "rot180", "rot270"}
    for d in result.per_transform_disagreement:
        assert d.mean_abs_deviation_from_ensemble_mean is not None
        assert d.inference_seconds >= 0


def test_per_transform_disagreement_is_none_when_members_not_kept():
    x = _input()
    result = run_stochastic_uncertainty(
        _biased_model, x, seed=42, band_names=BANDS, keep_per_member_predictions=False
    )
    for d in result.per_transform_disagreement:
        assert d.mean_abs_deviation_from_ensemble_mean is None


def test_metadata_records_seed_n_transforms_and_bands():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=7, band_names=BANDS)
    assert result.metadata["seed"] == 7
    assert result.metadata["n"] == 6
    assert result.metadata["band_names"] == list(BANDS)
    assert result.metadata["transform_names"] == [
        "identity", "hflip", "vflip", "rot90", "rot180", "rot270",
    ]


def test_band_names_length_must_match_input_channel_count():
    x = _input()
    with pytest.raises(ShapeMismatchError):
        run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=("B04", "B08"))


def test_empty_transforms_is_rejected():
    x = _input()
    with pytest.raises(InvalidEnsembleConfigError):
        run_stochastic_uncertainty(_biased_model, x, transforms=(), seed=42, band_names=BANDS)


def test_n_equals_one_is_well_defined():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, transforms=(IDENTITY,), seed=42, band_names=BANDS)
    assert result.n == 1
    assert result.scalar_summary == pytest.approx(0.0, abs=1e-6)


def test_visualization_normalization_never_changes_the_scientific_values():
    x = _input()
    result = run_stochastic_uncertainty(_biased_model, x, seed=42, band_names=BANDS)
    overall_std_map = result.std_prediction.mean(dim=0)
    before = result.overall_distribution.mean
    _ = normalize_for_visualization(overall_std_map)  # display-only transform
    after = result.overall_distribution.mean
    assert before == after  # the report's own stats are untouched by visualization scaling
    # and the map used to derive the report's own stats is itself unaffected by normalization
    assert overall_std_map.mean().item() == pytest.approx(before, abs=1e-4)


def test_same_seed_gives_numerically_identical_reports():
    x = _input()
    a = run_stochastic_uncertainty(_biased_model, x, seed=99, band_names=BANDS)
    b = run_stochastic_uncertainty(_biased_model, x, seed=99, band_names=BANDS)
    assert torch.equal(a.mean_prediction, b.mean_prediction)
    assert torch.equal(a.std_prediction, b.std_prediction)
    assert a.scalar_summary == b.scalar_summary
