"""Tests for frame.analysis.report -- the top-level `run_ndvi_analysis`
orchestrator (Phase 6), tying indices + comparison + uncertainty_overlay
together and attaching the four required scientific caveats.
"""

import numpy as np
import torch
import pytest

from frame.consistency import ComputationStatus
from frame.analysis.report import NDVIAnalysisReport, run_ndvi_analysis

BANDS = ("B04", "B03", "B02", "B08")
SCALE = 4
H = 8


def _reflectance(red, green, blue, nir, h=H):
    return torch.stack([torch.full((h, h), v) for v in (red, green, blue, nir)])


def test_report_structure():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)
    sr_std = torch.full((4, H * SCALE, H * SCALE), 0.01)
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert isinstance(report, NDVIAnalysisReport)
    assert report.native_ndvi.grid_label == "native_10m"
    assert report.sr_ndvi.grid_label == "sr_2_5m"
    assert report.native_ndvi.resolution_m == 10.0
    assert report.sr_ndvi.resolution_m == 2.5


def test_perfect_agreement_between_native_and_sr_gives_zero_comparison_error():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)  # SR "recovers" the exact same reflectance
    sr_std = torch.full((4, H * SCALE, H * SCALE), 0.0)
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert report.ndvi_comparison.comparison.status == ComputationStatus.COMPUTABLE
    assert report.ndvi_comparison.comparison.mean_abs_difference == pytest.approx(0.0, abs=1e-5)
    assert report.ndvi_comparison.comparison.rmse == pytest.approx(0.0, abs=1e-5)


def test_known_offset_between_native_and_sr_reflectance_produces_a_measurable_ndvi_difference():
    lr = _reflectance(red=0.1, green=0.0, blue=0.0, nir=0.3)  # NDVI = 0.5
    sr_mean = _reflectance(red=0.1, green=0.0, blue=0.0, nir=0.1, h=H * SCALE)  # NDVI = 0.0
    sr_std = torch.full((4, H * SCALE, H * SCALE), 0.02)
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert report.ndvi_comparison.comparison.mean_abs_difference == pytest.approx(0.5, abs=1e-4)


def test_uncertainty_maps_are_aligned_to_their_respective_grids():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)
    sr_std = torch.rand(4, H * SCALE, H * SCALE) * 0.01
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert report.uncertainty_overall_sr_grid.shape == (H * SCALE, H * SCALE)
    assert report.uncertainty_overall_native_grid.shape == (H, H)


def test_uncertainty_weighted_summary_is_populated():
    torch.manual_seed(0)
    lr = torch.rand(4, H, H) * 0.4 + 0.1
    sr_mean = torch.rand(4, H * SCALE, H * SCALE) * 0.4 + 0.1
    sr_std = torch.rand(4, H * SCALE, H * SCALE) * 0.01
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert report.uncertainty_weighted_summary.status == ComputationStatus.COMPUTABLE


def test_four_required_scientific_caveats_are_present():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)
    sr_std = torch.zeros(4, H * SCALE, H * SCALE)
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert len(report.scientific_caveats) == 4
    joined = " ".join(report.scientific_caveats).lower()
    assert "not a directly observed native 2.5" in joined or "not a directly observed native" in joined
    assert "not independent ground truth" in joined
    assert "not proof of physical accuracy" in joined
    assert "not a calibrated probability" in joined


def test_metadata_records_formula_and_resampling_and_bands():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)
    sr_std = torch.zeros(4, H * SCALE, H * SCALE)
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert "NDVI" in report.metadata["ndvi_formula"]
    assert report.metadata["resampling_method"] == "area_average_pool"
    assert report.metadata["band_names"] == list(BANDS)
    assert report.metadata["scale_factor"] == SCALE
    assert report.metadata["native_resolution_m"] == 10.0
    assert report.metadata["sr_resolution_m"] == 2.5


def test_deterministic_across_repeated_calls():
    torch.manual_seed(3)
    lr = torch.rand(4, H, H) * 0.4 + 0.1
    sr_mean = torch.rand(4, H * SCALE, H * SCALE) * 0.4 + 0.1
    sr_std = torch.rand(4, H * SCALE, H * SCALE) * 0.01
    a = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    b = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert torch.equal(a.native_ndvi.ndvi, b.native_ndvi.ndvi)
    assert a.ndvi_comparison.comparison.rmse == b.ndvi_comparison.comparison.rmse


def test_common_grid_mask_matches_comparison_valid_pixel_count():
    torch.manual_seed(4)
    lr = torch.rand(4, H, H) * 0.4 + 0.1
    sr_mean = torch.rand(4, H * SCALE, H * SCALE) * 0.4 + 0.1
    sr_std = torch.rand(4, H * SCALE, H * SCALE) * 0.01
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE)
    assert report.common_grid_mask.shape == (H, H)
    assert int(report.common_grid_mask.sum()) == report.ndvi_comparison.comparison.valid_pixel_count


def test_lr_mask_is_respected_when_given():
    lr = _reflectance(0.1, 0.2, 0.05, 0.5)
    sr_mean = _reflectance(0.1, 0.2, 0.05, 0.5, h=H * SCALE)
    sr_std = torch.zeros(4, H * SCALE, H * SCALE)
    lr_mask = np.ones((H, H), dtype=bool)
    lr_mask[0, 0] = False
    report = run_ndvi_analysis(lr, sr_mean, sr_std, band_names=BANDS, scale_factor=SCALE, lr_mask=lr_mask)
    assert report.native_ndvi.valid_mask[0, 0].item() is False
    assert report.ndvi_comparison.comparison.valid_pixel_count == H * H - 1
