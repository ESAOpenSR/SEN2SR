# NDVI downstream demonstration — Phase 6 experiment

Runs `frame.analysis.run_ndvi_analysis` (see `frame/analysis/README.md`) on
the **exact same deterministic Baseline 0 / Phase 5 scene** every prior
phase uses — no new scene, no re-fetch, no model re-inference.

This experiment does **not** modify `sen2sr/`, Baseline 0, or any Phase
1–5 artifact, and trains nothing.

## Scientific framing — read before the numbers below

> The purpose of this analysis is NOT to prove that 2.5 m NDVI is ground
> truth. It demonstrates that FRAME can provide a finer-grained
> vegetation-index visualization while explicitly exposing model-stability
> uncertainty.

1. SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid, not
   a directly observed native 2.5 m vegetation measurement.
2. The native 10 m NDVI and SR-derived NDVI are not independent ground
   truths — the SR output was itself derived from the same underlying
   10 m observation.
3. Agreement or disagreement between them measures internal consistency
   and downstream utility, not proof of physical accuracy.
4. Uncertainty here is the Phase 5 relative model-stability proxy, not a
   calibrated probability of NDVI error.

## No re-inference — reused artifacts

This experiment loads three tensors already saved by prior phases, instead
of re-fetching the scene from STAC or re-running the model:

| Reused artifact | Source phase |
|---|---|
| `experiments/baseline/outputs/input_tensor.pt` | Phase 0 — native 10 m LR reflectance |
| `experiments/uncertainty/outputs/sr_mean_tensor.pt` | Phase 5 — mean SR prediction (6-pass TTA ensemble) |
| `experiments/uncertainty/outputs/sr_std_tensor.pt` | Phase 5 — per-band model-stability uncertainty |

The script cross-checks scene identity (AOI, date window, time index)
between Baseline 0's and Phase 5's own saved metadata **before** trusting
them, and refuses to proceed on a mismatch.

## NDVI formula and resampling method

- `NDVI = (NIR − Red) / (NIR + Red)`, NIR = B08, Red = B04.
- The SR-grid NDVI is **never** compared directly against the native-grid
  NDVI — it is reduced to the native 10 m grid first via area-average
  pooling (`frame.consistency.downsample_to_lr_grid`, Phase 3, reused
  unchanged), the same method already used throughout this project for
  SR-to-LR grid reduction.

## Result of the last run — real measurements

| | |
|---|---:|
| Native NDVI valid pixels | 16,384 (128×128, full coverage) |
| SR NDVI valid pixels | 262,144 (512×512, full coverage) |
| Comparison (common-grid) valid pixels | 16,384 |
| Mean absolute NDVI difference | **0.00508** |
| RMSE | 0.00693 |
| Max absolute NDVI difference | 0.05794 |
| Correlation(uncertainty, \|ΔNDVI\|) | **+0.325** |
| Uncertainty-weighted mean \|ΔNDVI\| | 0.00594 |
| Unweighted mean \|ΔNDVI\| | 0.00508 |
| Runtime | 1.82 s (no model inference, no network) |

### Reading these numbers honestly

- The native-vs-downsampled-SR NDVI agreement is close (mean absolute
  difference ~0.005 on a [-1, 1] scale) — consistent with, and roughly the
  same order of magnitude as, Phase 3's own reflectance-level
  downsample-consistency RMSE on this identical scene. This is expected: it
  reflects the same underlying self-consistency the Fourier hard constraint
  already targets, not a new, independent validation.
- **Uncertainty correlates positively (r ≈ 0.32) with NDVI disagreement.**
  This is a genuine, moderate — not strong — positive relationship: pixels
  where the model's own predictions are less stable under TTA do tend to
  show somewhat larger NDVI disagreement between the native and downsampled-
  SR views. The uncertainty-weighted mean difference (0.00594) is
  correspondingly higher than the plain mean (0.00508), consistent with
  that correlation. This is reported as an observed, moderate correlation
  on one scene — not evidence of a calibrated or strong relationship, and
  not generalized beyond this run.
- Visually (`outputs/comparison_side_by_side.png`,
  `outputs/ndvi_absolute_difference.png`), the disagreement concentrates
  along field boundaries and roads — the same spatially-structured pattern
  Phase 5's own uncertainty map shows on this scene, not scattered noise.

## Layout

```
experiments/analysis/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/
    ndvi_native_10m.png                    # native 10 m NDVI
    ndvi_sr_2_5m.png                        # "SR-derived NDVI — 2.5 m pixel grid"
    ndvi_downsampled_sr_vs_native.png       # both on the SAME (native) grid, side by side
    ndvi_absolute_difference.png            # |ΔNDVI| on the native grid
    sr_ndvi_uncertainty_overlay.png         # SR NDVI + continuous uncertainty overlay (not binary)
    ndvi_histogram.png                      # NDVI value distributions, native vs. downsampled SR
    comparison_side_by_side.png             # RGB + native NDVI + SR NDVI + |ΔNDVI|, one summary figure
  metadata/
    run_metadata.json    # scene identity, formula, resampling method, all metrics, caveats, timestamps
```

## Running it

Same environment as Phases 0–5 (`sen2sr_venv/`). No new dependency was
added — `frame.analysis` uses only `torch` and `numpy`, both already
required. Requires Phase 0 (`experiments/baseline/run_baseline.py`) and
Phase 5 (`experiments/uncertainty/run_experiment.py`) to have already been
run at least once, so their output tensors exist to reuse.

```bash
sen2sr_venv/bin/python experiments/analysis/run_experiment.py
```

No internet access and no GPU are required — this experiment performs no
model inference and no data fetch at all.
