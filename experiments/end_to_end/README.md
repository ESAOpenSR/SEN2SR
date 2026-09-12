# End-to-end pipeline — Phase 9 experiment

Runs the complete FRAME pipeline in one real, sequential pass on the SAME
deterministic scene used throughout Phases 0–8:

```
input -> preprocessing -> frozen SEN2SR -> geospatial export
      -> uncertainty -> consistency diagnostics -> NDVI analysis
      -> final artifacts
```

Every stage calls the existing, already-tested `frame.*` modules directly
(`frame.preprocessing`, `frame.geospatial`, `frame.uncertainty`,
`frame.consistency`, `frame.analysis`) — the same functions
`frame/api/services/pipeline.py` orchestrates for the live API. Nothing
here reimplements Phase 1–8 science, `sen2sr/` is not modified, and
nothing is trained.

## Why this reuses the saved Baseline 0 tensor instead of re-fetching

Every prior phase's experiment (`baseline`, `baseline_preprocessing`,
`consistency`, `uncertainty`) re-queries `cubo.create(...)` with the same
AOI/date-window/band request, on the assumption that the STAC catalog
returns the same single scene every time for that historical window.

**That assumption was checked while building this phase and found false.**
Re-running the identical query today returns **3** time entries for the
same window — Phases 0–8 observed and indexed only **1** (`SCENE_TIME_INDEX
= 0`, documented as "the only-distinct item" at the time). The catalog has
drifted (most likely newly reprocessed/ingested scenes for that historical
date range), so a fresh fetch can no longer be trusted to return the same
scene at index 0 — re-fetching risks *silently substituting a different
scene*, which this phase's instructions explicitly forbid.

The only way to guarantee the literal same scene is to reuse the exact
tensor Baseline 0 already fetched and saved:
`experiments/baseline/outputs/input_tensor.pt`. Its geospatial context is
cross-checked against `experiments/baseline_geoexport/metadata/run_metadata.json`'s
`input_geospatial_metadata` (captured from the same original fetch, before
any tensor conversion), not re-derived.

## A note on pixel scale (found and fixed here, and in Phase 8)

This saved tensor's values are already in the 0–1 reflectance range
(verified: min ≈0.068, max ≈0.72, mean ≈0.23) — **not** raw digital
numbers. This script passes `input_scale="reflectance"` accordingly.
Passing `"raw_digital_number"` would silently re-divide already-scaled
reflectance and produce a near-zero, meaningless SR output with **no error
raised** — this exact failure mode was found during Phase 8 (the frontend
upload flow hit it first; the fix there was a UI toggle) and confirmed
again here at the experiment level.

## Running it

```bash
sen2sr_venv/bin/python experiments/end_to_end/run_experiment.py
```

Optional `--outputs-dir DIR --metadata-dir DIR` let a second, independent
run (the reproducibility check below) write elsewhere without overwriting
the canonical committed run.

## Outputs

| File | Description |
|---|---|
| `outputs/sr_mean.tif` | SR-derived product — 2.5 m pixel grid, 4 bands (B04/B03/B02/B08) |
| `outputs/uncertainty.tif` | Relative model-stability uncertainty, 5 bands (4 per-band std + `overall_std`), same grid as `sr_mean.tif` |
| `outputs/ndvi_native.tif` | Native 10 m NDVI, single band |
| `outputs/ndvi_sr.tif` | SR-derived NDVI — 2.5 m pixel grid, single band |
| `outputs/ndvi_diff.tif` | \|SR NDVI downsampled to the native grid − native NDVI\|, native grid, single band |
| `outputs/summary.png` | Concise 2×3 preview: native/SR RGB, uncertainty overlay, native/SR/diff NDVI |
| `metadata/run_metadata.json` | Full provenance: input identity, model identity, device, per-stage timings, shapes, CRS/transform/bounds, uncertainty configuration, NDVI statistics, self-consistency metrics, a cross-reference to the Phase 4 external-reference benchmark, git commit |

Every GeoTIFF is independently re-read and shape/band/CRS-checked before
the script reports success (`[verify]` lines) — write correctness is
never assumed.

## Measured results (this machine, this run)

| Stage | Time |
|---|---|
| Model load (cached weights) | 0.92 s |
| Preprocessing | 0.0005 s |
| Frozen SEN2SR + uncertainty (6-member TTA ensemble) | 0.29 s |
| Geospatial export (2 GeoTIFFs) | 0.01–0.02 s |
| Consistency diagnostics | 0.002 s |
| NDVI analysis + 3 GeoTIFFs | 0.01 s |
| Summary visualization (matplotlib) | ~1.1 s |
| **Total (excluding model load, which is cached across API requests in production)** | **≈2.3–2.4 s** |

Key numbers (see `metadata/run_metadata.json` for the full record):

- Uncertainty scalar summary: `1.174e-3` (mean per-pixel std across the
  6-member TTA ensemble, averaged over bands); distribution p95 = `2.53e-3`,
  max = `2.18e-2`.
- Self-consistency downsample RMSE (SR vs. its own LR input, own grid):
  `3.28e-3` (normalized RMSE `1.43e-2`).
- NDVI comparison (native vs. SR-derived, common native grid): mean
  absolute difference `5.08e-3`, RMSE `6.93e-3`, over all 16,384 valid
  native-grid pixels.

**None of these numbers are a ground-truth accuracy claim** — see
`docs/FINAL_SCIENTIFIC_AUDIT.md` for the full framing.

## Reproducibility

Run twice with identical configuration (same seed, same device, same
cached weights, same input tensor):

```bash
sen2sr_venv/bin/python experiments/end_to_end/run_experiment.py
sen2sr_venv/bin/python experiments/end_to_end/run_experiment.py --outputs-dir /tmp/run2/outputs --metadata-dir /tmp/run2/metadata
```

**Result, actually verified (not assumed) on this machine (NVIDIA GeForce
RTX 3050 Laptop GPU, CUDA):**

- Every metadata field except timestamps, per-stage timings, and
  output-path strings was **identical** across both runs (input identity,
  model identity, device, uncertainty configuration, shapes, geospatial
  fields, self-consistency metrics, NDVI metrics, environment).
- All 5 GeoTIFFs (`sr_mean.tif`, `uncertainty.tif`, `ndvi_native.tif`,
  `ndvi_sr.tif`, `ndvi_diff.tif`) were **bit-for-bit numerically identical**
  between the two runs (`numpy.array_equal` including NaN positions;
  max/mean absolute difference `0.0` on every array) — not just "within
  tolerance."
- Geospatial fields (CRS, affine transform, bounds) were identical between
  runs on every raster.

This bit-for-bit result is a real, measured outcome of this specific
model (frozen, no dropout, no batch-norm running-stats update at
inference), this specific fixed TTA transform set and seed, and this
specific GPU/driver/PyTorch build — it is **not** a general guarantee of
bit-identical GPU floating-point results across different hardware,
drivers, or PyTorch/CUDA versions, which is a well-known source of
non-determinism in floating-point accumulation order. What Phase 9
demonstrates is only what was actually run: on this machine, twice, the
result did not differ by even one bit.

## Scientific framing

Same as every prior phase — restated here because this is the one script
that exercises every stage together:

1. The SR output is a learned statistical inference resampled onto a
   **2.5 m pixel grid**, not a native Sentinel-2 2.5 m observation — Sentinel-2's
   finest native band resolution is 10 m.
2. Uncertainty is a **relative model-stability proxy** from test-time
   perturbation ensembling — not a calibrated probability of error, not a
   confidence interval, and not the upstream LAM explainability tool.
3. Self-consistency diagnostics compare the SR output against its own LR
   input only — not a ground-truth accuracy check.
4. NDVI agreement between the native and SR-derived grids measures
   internal consistency and downstream utility, not proof of physical
   accuracy — both are derived from the same underlying 10 m observation.
5. The Phase 4 `opensr-test` benchmark (cross-referenced in this
   experiment's metadata, not recomputed) evaluates the same frozen model
   against an **independent, different scene and sensor** (SPOT reference
   imagery) — it is not native Sentinel-2 2.5 m ground truth for this
   scene.

See `docs/FINAL_SCIENTIFIC_AUDIT.md` for the complete Phase 9 scientific
audit.
