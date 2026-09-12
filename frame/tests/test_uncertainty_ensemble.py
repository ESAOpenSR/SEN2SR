"""Tests for frame.uncertainty.ensemble -- the core N-pass TTA loop.

Uses small, fully deterministic fake "models" (plain callables following the
exact `model(x[None]) -> y[None]` convention every real sen2sr call in this
project uses) -- no network, no real SEN2SRLite weights.
"""

import torch
import torch.nn.functional as F
import pytest

from frame.uncertainty.ensemble import EnsembleRunResult, run_tta_ensemble
from frame.uncertainty.errors import InvalidEnsembleConfigError, InvalidTransformError
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS, HFLIP, IDENTITY, ROT90, VFLIP, Transform

SCALE = 4


def _equivariant_model(x_batched: torch.Tensor) -> torch.Tensor:
    """A perfectly geometry-equivariant 'model': plain bicubic upsampling.
    Flipping/rotating the input and undoing it on the output must give
    EXACTLY the same result as not transforming at all -- so an ensemble
    built from this model should show zero disagreement across transforms.
    """
    return F.interpolate(x_batched, scale_factor=SCALE, mode="bicubic", antialias=True)


def _biased_model(x_batched: torch.Tensor) -> torch.Tensor:
    """A deliberately NON-equivariant 'model': bicubic upsampling plus a
    bias derived from the input's own corner pixel -- which pixel sits in
    the corner differs across our geometric transforms, so this model's
    predictions genuinely disagree across ensemble members (a controlled,
    reproducible way to exercise 'variance is positive when predictions differ').
    """
    base = F.interpolate(x_batched, scale_factor=SCALE, mode="bicubic", antialias=True)
    bias = x_batched[0, 0, 0, 0].item() * 0.5
    return base + bias


def _input(seed=0, c=4, h=8, w=8):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, w, generator=g) * 0.5 + 0.1


# ---------------------------------------------------------------------------
# Shape / structure
# ---------------------------------------------------------------------------

def test_result_shape_matches_scale_factor_and_band_count():
    x = _input(h=8, w=8)
    result = run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert isinstance(result, EnsembleRunResult)
    assert result.mean_prediction.shape == (4, 32, 32)
    assert result.std_prediction.shape == (4, 32, 32)
    assert result.variance_prediction.shape == (4, 32, 32)


def test_n_and_transform_names_are_recorded():
    x = _input()
    result = run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert result.n == 6
    assert result.transform_names == ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


# ---------------------------------------------------------------------------
# Variance behavior -- the core scientific claims under test
# ---------------------------------------------------------------------------

def test_variance_is_zero_for_a_perfectly_equivariant_model():
    x = _input()
    result = run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert torch.allclose(result.variance_prediction, torch.zeros_like(result.variance_prediction), atol=1e-4)


def test_variance_is_positive_for_a_non_equivariant_model():
    x = _input()
    result = run_tta_ensemble(_biased_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert torch.any(result.variance_prediction > 1e-8)


def test_n_equals_one_gives_exactly_zero_variance_and_mean_equals_the_single_prediction():
    x = _input()
    result = run_tta_ensemble(_biased_model, x, (IDENTITY,), seed=42)
    assert result.n == 1
    assert torch.allclose(result.variance_prediction, torch.zeros_like(result.variance_prediction))
    with torch.no_grad():
        expected = _biased_model(x[None]).squeeze(0)
    assert torch.allclose(result.mean_prediction, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_and_config_gives_numerically_identical_results():
    x = _input()
    a = run_tta_ensemble(_biased_model, x, DEFAULT_TRANSFORMS, seed=42)
    b = run_tta_ensemble(_biased_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert torch.equal(a.mean_prediction, b.mean_prediction)
    assert torch.equal(a.variance_prediction, b.variance_prediction)
    assert torch.equal(a.std_prediction, b.std_prediction)


def test_seed_is_recorded_on_the_result():
    x = _input()
    result = run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=123)
    assert result.seed == 123


# ---------------------------------------------------------------------------
# Per-member predictions (kept for "which transform disagrees most")
# ---------------------------------------------------------------------------

def test_per_member_predictions_are_kept_when_requested():
    x = _input()
    result = run_tta_ensemble(_biased_model, x, DEFAULT_TRANSFORMS, seed=42, keep_per_member_predictions=True)
    assert result.per_member_predictions is not None
    assert len(result.per_member_predictions) == 6
    for pred in result.per_member_predictions:
        assert pred.shape == result.mean_prediction.shape


def test_per_member_predictions_are_discarded_when_not_requested():
    x = _input()
    result = run_tta_ensemble(_biased_model, x, DEFAULT_TRANSFORMS, seed=42, keep_per_member_predictions=False)
    assert result.per_member_predictions is None
    # the streaming stats must still be correct even without keeping members
    assert torch.any(result.variance_prediction > 1e-8)


def test_per_member_inference_seconds_recorded_for_every_member():
    x = _input()
    result = run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=42)
    assert len(result.per_member_inference_seconds) == 6
    assert all(s >= 0 for s in result.per_member_inference_seconds)
    assert result.total_seconds >= 0


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------

def test_empty_transform_list_is_rejected():
    x = _input()
    with pytest.raises(InvalidEnsembleConfigError):
        run_tta_ensemble(_equivariant_model, x, (), seed=42)


def test_a_broken_transform_is_rejected_before_any_inference():
    broken = Transform(name="broken", forward=lambda t: t.flip(-1), inverse=lambda t: t)
    x = _input()
    calls = {"n": 0}

    def counting_model(xb):
        calls["n"] += 1
        return _equivariant_model(xb)

    with pytest.raises(InvalidTransformError):
        run_tta_ensemble(counting_model, x, (IDENTITY, broken), seed=42)
    assert calls["n"] == 0  # validated before any (expensive) inference call


def test_rejects_non_3d_input_tensor():
    x = torch.rand(8, 8)
    with pytest.raises(InvalidTransformError):
        run_tta_ensemble(_equivariant_model, x, DEFAULT_TRANSFORMS, seed=42)
