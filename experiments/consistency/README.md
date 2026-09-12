# Consistency diagnostics — Phase 3 experiment

Runs `frame.consistency.run_consistency_diagnostics` (see
`frame/consistency/README.md`) against a real SR result, produced from the
**exact same deterministic Baseline 0 scene** (same AOI, same date window,
same time index, same bands, same unmodified `SEN2SRLite/NonReference_RGBN_x4`
model), preprocessed by the Phase 1 `frame.preprocessing` layer and
geospatially described the same way Phase 2 (`frame.geospatial`) established.

This experiment does **not** modify `experiments/baseline/`,
`experiments/baseline_preprocessing/`, or `experiments/baseline_geoexport/`
in any way.

## What this proves

That `frame.consistency`'s diagnostics run end-to-end against a real model
output and produce real, reproducible numbers.

## What this does NOT prove

**Nothing about accuracy.** Every number below is a **self-consistency**
diagnostic — how well the SR output, reduced back to its LR grid, agrees
with the real LR observation it was derived from — never a comparison
against an independent higher-resolution reference (none exists for
Sentinel-2; see `docs/FRAME_TECHNICAL_SPEC.md` Section 11). Nothing here is
labeled "accuracy," and no numeric pass/fail threshold is applied anywhere —
see `frame/consistency/README.md`'s Threshold Policy section.

## Result of the last run

All diagnostics were `COMPUTABLE` (mask coverage 1.0 — no clouds/nodata in
this scene, so every one of the 16,384 pixels contributed):

| Diagnostic | Value |
|---|---|
| Downsample-consistency — overall RMSE | 0.00330 (reflectance units) |
| Downsample-consistency — overall mean abs. error | 0.00223 |
| Downsample-consistency — overall normalized RMSE | 0.0143 (RMSE ÷ mean \|LR\|) |
| Downsample-consistency — overall max abs. error | 0.0410 |
| Per-band RMSE — B04 (Red) | 0.00299 |
| Per-band RMSE — B03 (Green) | 0.00324 |
| Per-band RMSE — B02 (Blue) | 0.00266 |
| Per-band RMSE — B08 (NIR) | 0.00413 |
| NDVI comparison — mean abs. discrepancy | 0.00489 |
| NDVI comparison — RMSE | 0.00665 |
| NDVI comparison — max abs. discrepancy | 0.0450 |
| B08/B04 ratio comparison — mean abs. discrepancy | 0.0222 |
| B08/B04 ratio comparison — RMSE | 0.0332 |
| B08/B04 ratio comparison — max abs. discrepancy | 0.711 |
| Cross-tile | `None` — see "Cross-tile limitation" below |

The full, exact numbers (plus every parameter needed to reproduce them) are
in `metadata/run_metadata.json`, under `consistency_diagnostics`;
re-running regenerates them.

### Reading these numbers honestly

- The per-band RMSEs (~0.003–0.004, on reflectance values roughly in
  [0, 1]) and the near-zero discrepancy maps (`outputs/*_comparison.png`)
  are consistent with the upstream Fourier hard constraint
  (`sen2sr/models/tricks.py`, unmodified) doing what it is designed to do:
  lock the SR output's low-frequency content to the real LR measurement.
  This is expected, not a novel finding — it is exactly the self-consistency
  diagnostic confirming the existing constraint held on this run.
- The B08/B04 ratio's `max_abs_discrepancy` (0.71) is far larger than its
  `mean_abs_discrepancy` (0.022) — visible in `outputs/b08_b04_ratio_comparison.png`
  as a single bright outlier pixel. This is the expected behavior of a
  *ratio* metric near a small denominator (a tiny absolute error in B04 near
  zero produces a large relative swing in the ratio) — exactly the kind of
  instability `docs/FRAME_TECHNICAL_SPEC.md` Section 12 warns SAM/ratio-style
  metrics are prone to, not evidence of a pipeline defect. It is reported
  raw, not smoothed away or thresholded out.

### Cross-tile limitation — explicit, not silently skipped

Baseline 0's scene is a single 128×128 patch (`EDGE_SIZE_PX` matches the
model's native patch size), so `sen2sr.utils.predict_large`'s multi-tile
path is never triggered — there is only ever one tile, so there is no
overlap region to measure. `frame.consistency.compare_tile_overlap` is
implemented and unit-tested (`frame/tests/test_consistency_tiles.py`,
synthetic tiles only), but this experiment correctly reports `cross_tile:
null` rather than fabricating a tile pair. See `frame/consistency/tiles.py`'s
module docstring for the verified reason `sen2sr.utils.predict_large` cannot
be wired to this today without modifying `sen2sr` (out of scope).

## How the experiment is structured

1. Fetch the **exact same deterministic scene** Baseline 0 uses (constants
   duplicated, not imported — same convention as
   `experiments/baseline_preprocessing/` and `experiments/baseline_geoexport/`).
2. Reuse Baseline 0's already-downloaded model weights cache — no re-download.
3. Extract geospatial metadata (CRS, transform, bounds) from the fetched
   `cubo` cube, exactly as Phase 1/Phase 2 do.
4. Run `frame.preprocessing.preprocess_rgbn(..., require_geospatial=True)`.
5. Run the real, unmodified upstream model.
6. Derive output geospatial metadata via `frame.geospatial.derive_output_metadata`
   (Phase 2) — recorded for traceability; this experiment does not write a
   new GeoTIFF, since `experiments/baseline_geoexport/` already proves that
   round trip.
7. Run `frame.consistency.run_consistency_diagnostics` (Phase 3, new) on the
   real LR tensor, SR tensor, and validity mask.
8. Save the JSON diagnostics report (`metadata/run_metadata.json`), and
   three diagnostic visualizations (`outputs/`):
   - `ndvi_comparison.png` — LR-derived vs. downsampled-SR-derived NDVI,
     plus their spatial difference.
   - `b08_b04_ratio_comparison.png` — same, for the B08/B04 ratio.
   - `spatial_discrepancy_map.png` — per-pixel mean absolute error across
     bands (downsample-consistency), showing *where* in the image the
     discrepancy concentrates.

## Layout

```
experiments/consistency/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/              # 3 diagnostic visualizations (created by the script)
  metadata/              # run_metadata.json (created by the script)
```

## Running it

Same environment as Baseline 0 / Phase 1 / Phase 2 (`sen2sr_venv/`). No new
dependency was added for this phase — `frame.consistency` uses only `numpy`
and `torch`, both already required.

From the repository root:

```bash
sen2sr_venv/bin/python experiments/consistency/run_experiment.py
```

Requires internet access (a live STAC endpoint) and the same model weights
cache Baseline 0 downloads (auto-downloaded if not already cached).
