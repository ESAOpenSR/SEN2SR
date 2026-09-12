# `frame.analysis`

Phase 6 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`
Section 15's agriculture demonstration — the first, lightest of the three
planned downstream analyses). Computes NDVI from the existing proven
4-band RGBN pipeline at both the native 10 m grid and the SR 2.5 m pixel
grid, compares them on a common grid, and relates that comparison to
Phase 5's model-stability uncertainty.

This package imports nothing from `sen2sr`, calls no model, and does not
even call `frame.uncertainty`'s ensemble loop — it consumes Phase 5's
already-computed `mean_prediction`/`std_prediction` directly. See
`experiments/analysis/run_experiment.py` for the real integration, which
reuses three already-saved tensors from Phases 0 and 5 rather than
re-fetching the scene or re-running the model.

## What this demonstrates — and what it does not

**The purpose is NOT to prove that 2.5 m NDVI is ground truth.** It
demonstrates that FRAME can provide a finer-grained vegetation-index
visualization while explicitly exposing model-stability uncertainty
alongside it. Every `NDVIAnalysisReport` carries four caveats verbatim
(`frame.analysis.report.SCIENTIFIC_CAVEATS`):

1. SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid, not
   a directly observed native 2.5 m vegetation measurement.
2. The native 10 m NDVI and SR-derived NDVI are not independent ground
   truths — the SR output was itself derived from the same underlying
   10 m observation.
3. Agreement or disagreement between them measures internal consistency
   and downstream utility, not proof of physical accuracy.
4. Uncertainty here is the Phase 5 relative model-stability proxy, not a
   calibrated probability of NDVI error.

## NDVI — formula and safe division

`NDVI = (NIR − Red) / (NIR + Red)`, where NIR = B08, Red = B04. The
division itself is **reused, not reimplemented** — `frame.analysis.indices`
calls `frame.consistency.spectral_ratios.compute_ndvi` (Phase 3, already
tested), which excludes any pixel where `|NIR + Red|` is too small to
divide by safely rather than emitting a runtime warning or a fabricated
value. Excluded pixels carry `NaN` in the `ndvi` array but are always
excluded from `valid_mask` — every valid pixel is guaranteed finite
(`frame/tests/test_analysis_indices.py::test_no_nan_or_inf_in_valid_pixels`).

## Never a 2.5 m-vs-10 m comparison directly

Per this phase's explicit instruction, `frame.analysis.comparison` never
compares the SR-grid NDVI against the native-grid NDVI at mismatched
resolutions. The SR NDVI is always reduced first via
`frame.consistency.downsample_to_lr_grid` (Phase 3's area-average pooling —
reused, this project's one documented "scientifically appropriate" method
for reducing an SR-grid raster to its parent grid), then compared pixel-for-
pixel against the native NDVI on that common grid.

## Relating uncertainty to NDVI disagreement — no invented threshold

`frame.analysis.uncertainty_overlay` aggregates Phase 5's 4-band
`std_prediction` to one map by averaging over bands — the exact convention
`experiments/uncertainty/run_experiment.py` already uses. It is then
downsampled to the same native grid as the NDVI comparison (same reused
downsampling function), and related to the NDVI absolute-difference map two
threshold-free ways:

- **Pearson correlation** between the uncertainty map and the absolute
  NDVI difference map — answers "does uncertainty track disagreement,
  roughly" without defining what counts as "high."
- **Uncertainty-weighted mean absolute difference** — the same mean
  absolute difference, but weighted by the uncertainty map itself, reported
  alongside the plain (unweighted) mean for direct comparison.

**No "high uncertainty" or "confidence" threshold is defined anywhere in
this package** — `frame/tests/test_analysis_uncertainty_overlay.py::test_summary_does_not_carry_a_threshold_field`
enforces this structurally, the same convention every prior `frame.*`
package in this project already follows.

## Visualization labeling

Every SR-grid NDVI plot is titled **"SR-derived NDVI — 2.5 m pixel
grid"**, never unqualified "2.5 m NDVI." Every uncertainty visualization is
titled **"Relative model-stability uncertainty"** and rendered as a
continuous, percentile-normalized overlay (`frame.uncertainty.statistics.normalize_for_visualization`,
reused) — never thresholded into a binary good/bad mask.

## Public API

```python
from frame.analysis import run_ndvi_analysis

report = run_ndvi_analysis(
    lr_reflectance,       # (4, H, W) -- native 10 m reflectance (frame.preprocessing output)
    sr_mean_prediction,   # (4, H*4, W*4) -- Phase 5's UncertaintyResult.mean_prediction
    sr_std_prediction,    # (4, H*4, W*4) -- Phase 5's UncertaintyResult.std_prediction
    band_names=("B04", "B03", "B02", "B08"),
    scale_factor=4,
)

report.native_ndvi.ndvi              # (H, W), native 10 m grid
report.sr_ndvi.ndvi                  # (H*4, W*4), SR 2.5 m pixel grid
report.ndvi_comparison.comparison    # mean_abs_difference, rmse, max_abs_difference, valid_pixel_count
report.uncertainty_weighted_summary  # correlation + weighted/unweighted mean abs diff
report.scientific_caveats            # the four caveats above, verbatim
```

## Tests

```bash
sen2sr_venv/bin/python -m pytest \
    frame/tests/test_analysis_indices.py \
    frame/tests/test_analysis_comparison.py \
    frame/tests/test_analysis_uncertainty_overlay.py \
    frame/tests/test_analysis_report.py \
    frame/tests/test_analysis_geospatial_integration.py -v
```

All 44 tests use small synthetic tensors — no network, no real model, no
real benchmark data. The geospatial integration test uses real
`frame.geospatial`/`frame.preprocessing` code with synthetic metadata.
