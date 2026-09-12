# Stochastic TTA uncertainty — Phase 5 experiment

Runs FRAME's proven, **unmodified** `SEN2SRLite/NonReference_RGBN_x4` model
through `frame.uncertainty.run_stochastic_uncertainty` (see
`frame/uncertainty/README.md`) on the **exact same deterministic Baseline 0
scene** every prior phase uses (same AOI, same date window, same time
index, same bands) — no new scene, no silent substitution.

This experiment does **not** modify `sen2sr/`, Baseline 0, or any Phase
1–4 artifact, and trains nothing.

## Scientific framing — read before the numbers below

> This uncertainty estimate measures prediction stability under the
> selected test-time perturbations. It is a relative model-stability signal
> and is not a calibrated probability of error, confidence interval, or
> physically rigorous uncertainty bound.

**This is not LAM.** LAM (`sen2sr/xai/lam.py`, upstream, unmodified) is
explainability/sensitivity — gradients against blurred copies of the input,
answering "which input pixels matter." This experiment is a stability/
uncertainty proxy — repeated forward passes on geometrically re-framed
copies of the same input, no gradients, answering "how much does the
model's own prediction disagree with itself." Neither substitutes for the
other; see `frame/uncertainty/README.md`'s comparison table.

**The 2.5 m pixel grid is still not a native observation.** A low
uncertainty value means the model is self-consistent here, not that this
pixel's reflectance was measured at 2.5 m — Sentinel-2 never did that
(docs/FRAME_TECHNICAL_SPEC.md Section 1.4).

## Exact ensemble configuration

| | |
|---|---|
| Model | `SEN2SRLite/NonReference_RGBN_x4` (unmodified) |
| Primary ensemble size | N = 6 (the full default geometric transform set) |
| Transforms | `identity, hflip, vflip, rot90, rot180, rot270` — every one has an exact forward/inverse pair; no hidden random state |
| Seed | 42 |
| Optional noise perturbation | **not used** — the default experiment uses deterministic geometric TTA alone, per this phase's explicit instruction |
| N-sweep | N = 1, 4, 8, 16 (N>6 cyclically repeats the 6 transforms — documented, not a bug) |

## Result of the last run — real measurements, not estimates

**Primary ensemble (N=6):**

| Statistic | Value |
|---|---:|
| `scalar_summary` (mean per-pixel std, band-averaged) | 0.001174 |
| median | 0.000993 |
| std (of the std map itself) | 0.000749 |
| min | 0.000104 |
| p90 | 0.002040 |
| p95 | 0.002526 |
| max | 0.021818 |

All values are in reflectance units (the SR output's own native scale,
roughly 0–1). The distribution is strongly right-skewed (see
`outputs/uncertainty_histogram.png`): most pixels sit near ~0.0005, with a
long tail out past 0.02 at a small number of pixels.

**Per-transform disagreement** (mean |prediction − ensemble mean|):

| Transform | Disagreement | Inference time |
|---|---:|---:|
| identity | 0.000981 | 0.145 s (includes one-time CUDA warmup) |
| hflip | 0.000921 | 0.012 s |
| vflip | 0.000927 | 0.009 s |
| rot90 | **0.001057** | 0.009 s |
| rot180 | 0.000983 | 0.009 s |
| rot270 | **0.001050** | 0.010 s |

The two 90°-type rotations show the largest disagreement from the ensemble
mean; the two flips show the smallest. One scene is not enough evidence to
generalize this pattern — reported as observed, not claimed as a general
property of the model.

**N-sweep** (`outputs/n_sweep.png`):

| N | scalar_summary | p95 | max |
|---:|---:|---:|---:|
| 1 | 0.000000 (well-defined: a single member has no dispersion) | 0.000000 | 0.000000 |
| 4 | 0.001065 | 0.002360 | 0.023978 |
| 8 | 0.001145 | 0.002463 | 0.023142 |
| 16 | 0.001166 | 0.002509 | 0.022509 |

The summary rises steeply from N=1 to N=4, then **plateaus** — N=8 and
N=16 (which cyclically repeat the same 6 transforms) land close to the true
N=6 value (0.001174) rather than continuing to grow or shrink toward zero.
This is the expected behavior for a *fixed, small, repeating* transform set
— it does not mean uncertainty "converges" in the way a genuinely
larger/richer ensemble might; it means repeating an already-seen transform
adds no new information, exactly as documented in
`frame/uncertainty/README.md`. Uncertainty is **not** monotonically
decreasing with N here, nor does it need to be.

## Where uncertainty is concentrated — answering the analysis questions

Read directly from `outputs/sr_and_uncertainty.png` and
`outputs/uncertainty_overlay.png`:

- **Non-zero and spatially structured, not uniform noise.** The uncertainty
  map has clear spatial pattern — it is not a flat, texture-free field.
- **Concentrated along field boundaries, roads, and the built-up area**
  (visible top-left of the scene) — exactly the "plausible difficult
  regions" (edges, texture, high-frequency structure) the analysis
  questions anticipated, not scattered at random.
- **Low uncertainty inside homogeneous field interiors** — large flat
  agricultural parcels show near-zero dispersion across every transform.
- No clouds are present in this scene (mask coverage 1.0, confirmed in
  `metadata/run_metadata.json`), so cloud-boundary behavior specifically is
  not exercised by this run.

## Layout

```
experiments/uncertainty/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/
    sr_mean.tif                       # SR mean prediction, GeoTIFF, co-registered
    uncertainty.tif                   # 5-band GeoTIFF: B04_std,B03_std,B02_std,B08_std,overall_std
    sr_mean_tensor.pt, sr_std_tensor.pt
    sr_and_uncertainty.png            # RGB mean + overall uncertainty map, side by side
    uncertainty_overlay.png           # RGB mean with uncertainty heatmap overlaid
    uncertainty_histogram.png         # distribution of per-pixel uncertainty (raw values)
    per_transform_disagreement.png    # which transform disagreed most
    n_sweep.png                       # scalar_summary vs. N
  metadata/
    run_metadata.json    # full config, statistics definitions, geospatial metadata, caveats, timestamps, git commit
```

## Running it

Same environment as Phases 0–4 (`sen2sr_venv/`). No new dependency was
added for this phase — `frame.uncertainty` uses only `torch` and `numpy`,
both already required.

```bash
sen2sr_venv/bin/python experiments/uncertainty/run_experiment.py
```

Requires internet access on first run (Hugging Face for weights, a live
STAC endpoint for the imagery — both already cached from prior phases on
this machine). GPU used automatically when available (measured: primary
ensemble negligible VRAM, consistent with Baseline 0's own ~100 MiB-peak
measurement for this model at this patch size).
