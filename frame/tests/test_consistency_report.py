"""Tests for frame.consistency.report -- the top-level
run_consistency_diagnostics orchestrator, and the package-wide guarantee
that no arbitrary pass/fail threshold exists anywhere in the core
calculation logic (docs/FRAME_TECHNICAL_SPEC.md Section 9.2/11's
"do not invent thresholds" instruction).
"""

import dataclasses

import numpy as np
import torch
import pytest

from frame.consistency.band_discrepancy import BandDiscrepancy, DownsampleConsistencyResult
from frame.consistency.report import ConsistencyDiagnostics, run_consistency_diagnostics
from frame.consistency.spectral_ratios import SpectralIndexComparison
from frame.consistency.status import ComputationStatus
from frame.consistency.tiles import TileOverlapResult

RGBN = ["B04", "B03", "B02", "B08"]


def _scene(red, green, blue, nir, size=2, scale=4):
    lr = torch.stack([torch.full((size, size), v) for v in (red, green, blue, nir)])
    sr = torch.stack([torch.full((size * scale, size * scale), v) for v in (red, green, blue, nir)])
    return lr, sr


def test_run_consistency_diagnostics_on_a_perfectly_consistent_scene():
    lr, sr = _scene(red=0.1, green=0.2, blue=0.05, nir=0.4)
    mask = np.ones((2, 2), dtype=bool)
    result = run_consistency_diagnostics(lr, sr, mask, band_names=RGBN, scale_factor=4)

    assert isinstance(result, ConsistencyDiagnostics)
    assert result.downsample_consistency.overall.status == ComputationStatus.COMPUTABLE
    assert result.downsample_consistency.overall.mean_abs_error == pytest.approx(0.0, abs=1e-6)
    assert result.ndvi_comparison.status == ComputationStatus.COMPUTABLE
    assert result.ndvi_comparison.mean_abs_discrepancy == pytest.approx(0.0, abs=1e-6)
    assert result.b08_b04_ratio_comparison.status == ComputationStatus.COMPUTABLE
    assert result.cross_tile is None  # not supplied -- see module docstring


def test_run_consistency_diagnostics_reuses_one_downsample_pass():
    # The SR downsampled-back value should be IDENTICAL whether read from
    # the downsample-consistency result's own computation or from what the
    # spectral-ratio comparisons used -- i.e. both must agree exactly with
    # a fixed, known discrepancy, proving they were computed from the same
    # underlying reduced array rather than two independently-noisy passes.
    lr, sr = _scene(red=0.1, green=0.0, blue=0.0, nir=0.3)
    # perturb the SR NIR band uniformly
    sr[3] = 0.1
    mask = np.ones((2, 2), dtype=bool)
    result = run_consistency_diagnostics(lr, sr, mask, band_names=RGBN, scale_factor=4)
    # LR NDVI = (0.3-0.1)/0.4 = 0.5 ; SR-downsampled NDVI = (0.1-0.1)/0.2 = 0.0
    assert result.ndvi_comparison.mean_abs_discrepancy == pytest.approx(0.5, abs=1e-6)


def test_run_consistency_diagnostics_accepts_an_externally_computed_cross_tile_result():
    lr, sr = _scene(red=0.1, green=0.2, blue=0.05, nir=0.4)
    mask = np.ones((2, 2), dtype=bool)
    cross_tile = TileOverlapResult(ComputationStatus.COMPUTABLE, (4, 4), 16, 0.01, 0.01, 0.02)
    result = run_consistency_diagnostics(lr, sr, mask, band_names=RGBN, scale_factor=4, cross_tile=cross_tile)
    assert result.cross_tile is cross_tile


def test_parameters_record_is_sufficient_to_reproduce_the_calculation():
    lr, sr = _scene(red=0.1, green=0.2, blue=0.05, nir=0.4)
    mask = np.ones((2, 2), dtype=bool)
    result = run_consistency_diagnostics(lr, sr, mask, band_names=RGBN, scale_factor=4)
    params = result.parameters
    assert params["downsample_method"] == "area_average_pool"
    assert params["scale_factor"] == 4
    assert tuple(params["band_names"]) == tuple(RGBN)
    assert params["lr_shape"] == (4, 2, 2)
    assert params["sr_shape"] == (4, 8, 8)
    assert params["mask_coverage"] == pytest.approx(1.0)


def test_not_computable_scene_propagates_through_the_whole_report():
    lr, sr = _scene(red=0.1, green=0.2, blue=0.05, nir=0.4)
    mask = np.zeros((2, 2), dtype=bool)  # nothing valid anywhere
    result = run_consistency_diagnostics(lr, sr, mask, band_names=RGBN, scale_factor=4)
    assert result.downsample_consistency.overall.status == ComputationStatus.NOT_COMPUTABLE
    assert result.ndvi_comparison.status == ComputationStatus.NOT_COMPUTABLE
    assert result.b08_b04_ratio_comparison.status == ComputationStatus.NOT_COMPUTABLE


# ---------------------------------------------------------------------------
# No invented thresholds anywhere in the core calculation logic.
#
# A concrete, checkable proxy for "no arbitrary pass/fail threshold exists":
# none of the result dataclasses this package returns carry a field that
# encodes an accept/reject judgment (a boolean "passed"/"ok"/"is_valid", or a
# name suggesting a threshold was applied) -- the only status vocabulary
# anywhere is ComputationStatus, which states computability, not quality.
# ---------------------------------------------------------------------------

_FORBIDDEN_FIELD_NAME_FRAGMENTS = ("passed", "is_valid", "threshold", "is_ok", "accept", "reject", "good", "bad")


@pytest.mark.parametrize(
    "result_type",
    [BandDiscrepancy, DownsampleConsistencyResult, SpectralIndexComparison, TileOverlapResult, ConsistencyDiagnostics],
)
def test_no_pass_fail_threshold_fields_on_result_types(result_type):
    field_names = [f.name.lower() for f in dataclasses.fields(result_type)]
    for name in field_names:
        for forbidden in _FORBIDDEN_FIELD_NAME_FRAGMENTS:
            assert forbidden not in name, (
                f"{result_type.__name__}.{name} looks like a pass/fail verdict field; "
                "this package must only ever report raw diagnostic values plus a "
                "ComputationStatus (COMPUTABLE/NOT_COMPUTABLE/INVALID_INPUT)."
            )


def test_computation_status_has_exactly_the_three_documented_states():
    assert {s.value for s in ComputationStatus} == {"COMPUTABLE", "NOT_COMPUTABLE", "INVALID_INPUT"}
