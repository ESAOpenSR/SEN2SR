# `frame.uncertainty`

Phase 5 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`
Section 13's recommendation: **stochastic inference / test-time
perturbation ensembling**, now implemented on top of the frozen,
unmodified `SEN2SRLite/NonReference_RGBN_x4` path).

This package imports nothing from `sen2sr` and calls no model itself — the
real model call happens in `experiments/uncertainty/run_experiment.py`,
matching the separation of concerns established in Phases 1–4.

## What this measures — and what it explicitly does not

**This uncertainty estimate measures prediction stability under the
selected test-time perturbations. It is a relative model-stability signal
and is not a calibrated probability of error, confidence interval, or
physically rigorous uncertainty bound.**

Concretely: the frozen model is run once per geometric transform (identity,
horizontal flip, vertical flip, and the three non-trivial 90°-multiple
rotations) on the *same real observation*, each transform is undone on the
model's output, and the per-pixel dispersion across those six aligned
predictions is reported. A pixel where the model agrees with itself across
every framing gets a low value; a pixel where re-framing the same input
changes what the model predicts gets a high value. There is no ground truth
anywhere in this computation — nothing here is calibrated against a known
error rate, and a low value does not prove the prediction is correct, only
that it is *stable* under this specific set of perturbations.

## TTA uncertainty vs. LAM — do not conflate these

| | LAM (`sen2sr/xai/lam.py`, upstream, unmodified) | TTA uncertainty (`frame.uncertainty`, this package) |
|---|---|---|
| Question answered | "Which input pixels most influence this output?" | "How much does the model's own prediction disagree with itself across equivalent views of the same input?" |
| Mechanism | Gradients (backprop) against progressively blurred copies of ONE input | Repeated forward passes (no gradients) on geometrically transformed copies of ONE input |
| Category | Explainability / sensitivity | Stability / uncertainty proxy |
| Output | A saliency (KDE) map, a Gini "complexity" scalar, a blur-robustness curve | A per-pixel mean/std/variance map, a scalar summary, a distribution of that map's values |

Both are legitimate, complementary diagnostics — neither is a substitute for
the other, and this package never calls its own output "LAM," and LAM is
never called "uncertainty," anywhere in this project.

## The 2.5 m pixel grid — still not a native observation

Nothing in this package changes docs/FRAME_TECHNICAL_SPEC.md Section 1.4's
distinction: the uncertainty raster describes confidence in a *statistical
inference* resampled onto a 2.5 m pixel grid, not confidence about a
2.5 m *sensor measurement* — Sentinel-2 never made one. A low uncertainty
value means "the model is self-consistent here," not "this pixel's
reflectance was actually observed at 2.5 m."

## Method — exactly what `run_stochastic_uncertainty` does

```
for each transform in {identity, hflip, vflip, rot90, rot180, rot270}:
    x_t   = transform.forward(input_tensor)      # re-frame the LR input
    y_t   = model(x_t[None]).squeeze(0)           # the SAME frozen model, unmodified
    y     = transform.inverse(y_t)                # undo the re-framing on the SR output
    accumulate(y)                                 # Welford streaming mean/variance

mean_prediction     = running mean across the N (de-transformed) predictions
variance_prediction = running POPULATION variance (divide by N, not N-1)
std_prediction      = sqrt(variance_prediction)
```

Every transform has an exact forward/inverse pair (`frame.uncertainty.transforms`);
`identity` is always included; none carries hidden random state. Statistics
are accumulated online (Welford's algorithm, `frame.uncertainty.statistics.WelfordAccumulator`)
so the full set of N predictions never needs to be held in memory at once
(`keep_per_member_predictions=False` discards them after each update; the
default keeps them only to support the per-transform disagreement
breakdown below, since at this project's proven 128×128 patch size an
individual prediction is a few MB — free to keep for N ≤ a few dozen).

## Statistics — exactly what each field means

- `mean_prediction`, `variance_prediction`, `std_prediction` — per-pixel,
  per-band, shape `(bands, H·scale, W·scale)`.
- `scalar_summary` — **the mean of the per-pixel standard deviation map,
  averaged over bands** (`frame.uncertainty.report.SCALAR_SUMMARY_DEFINITION`
  states this explicitly on every result — never left ambiguous which
  single-number statistic it is).
- `overall_distribution` / `per_band_distribution` — the *distribution* of
  that per-pixel std map's own values: `mean`, `median`, `std`, `p90`,
  `p95`, `min`, `max`. No percentile or moment here is elevated to a
  pass/fail threshold — `frame/tests/test_uncertainty_statistics.py::test_distribution_stats_does_not_carry_a_pass_fail_threshold_field`
  enforces this structurally, the same convention `frame.consistency` and
  `frame.validation` already established.
- `per_transform_disagreement` — for each transform, its own prediction's
  mean absolute deviation from the final ensemble mean, plus its inference
  time — answers "which transformation produced the largest disagreement."

## Visualization normalization — display only, never scientific

`frame.uncertainty.statistics.normalize_for_visualization` percentile-clips
and rescales a map to `[0, 1]` **for rendering only**. It is never applied
before computing `mean_prediction`/`variance_prediction`/`std_prediction`
or any distribution statistic, and its output must never be saved as, or
mistaken for, the actual uncertainty values — see
`experiments/uncertainty/README.md`'s visualizations, each labeled
**"Relative model-stability uncertainty (higher = less stable under
TTA)"**, never with unexplained color semantics like "red = bad."

## Optional noise perturbation — a documented proposal, not the default

The Phase 5 task allows an *optional*, small-amplitude reflectance-noise
perturbation "only if scientifically justified." **It is not implemented
as part of `DEFAULT_TRANSFORMS` and the default experiment does not use
it.** If added later, it would need a different `inverse` semantics than
the geometric transforms above: additive input noise has no meaningful
"undo" on the SR output (there is nothing geometric to reverse), so its
`inverse` would be the identity function on the output, and the ensemble
would instead be measuring dispersion under small *input reflectance*
perturbations — a genuinely different (and weaker-justified, since the
noise amplitude itself would be a free parameter with no principled value
yet) uncertainty notion than the geometry-based one this phase implements.
Left as a documented extension point, not built.

## Threshold policy

**No numeric "high uncertainty" threshold is defined anywhere in this
package.** Every distribution statistic above is reported raw; interpreting
"how much dispersion is a lot" is left to the experiment's written analysis
(`experiments/uncertainty/README.md`), which characterizes the *empirical*
distribution observed on the one deterministic scene evaluated so far — not
enough evidence to fix a threshold, the same reasoning `frame.consistency`
and `frame.validation` already apply.

## Geospatial co-registration

`frame.uncertainty` never imports `frame.geospatial` and writes no files —
by design, the same separation Phases 1–4 use. The uncertainty raster is
made co-registered with its SR raster entirely by **reuse**: both are
written with the exact same `frame.geospatial.derive_output_metadata(...)`
result (same CRS, affine transform, bounds, width, height), at the
experiment level (`experiments/uncertainty/run_experiment.py`). No new
geospatial logic was written for this phase —
`frame/tests/test_uncertainty_geospatial_integration.py` proves the pattern
holds by writing and independently re-reading both a real SR GeoTIFF and a
real uncertainty GeoTIFF and checking every geospatial field matches.

## Public API

```python
from frame.uncertainty import run_stochastic_uncertainty, DEFAULT_TRANSFORMS

result = run_stochastic_uncertainty(
    model,                        # model(x[None]) -> y, the same calling convention as every prior phase
    lr_tensor,                    # (C, H, W)
    transforms=DEFAULT_TRANSFORMS,  # optional; this is the default
    seed=42,
    band_names=("B04", "B03", "B02", "B08"),
)

result.mean_prediction        # (C, H*4, W*4)
result.std_prediction         # (C, H*4, W*4)
result.scalar_summary         # float -- mean per-pixel std, averaged over bands
result.overall_distribution   # mean/median/std/p90/p95/min/max of the std map
result.per_transform_disagreement  # which transform disagreed most
```

Raises a specific `UncertaintyError` subclass for structurally invalid
input:

| Problem | Exception |
|---|---|
| Empty transform list | `InvalidEnsembleConfigError` |
| A transform that doesn't round-trip (forward→inverse ≠ identity) | `InvalidTransformError` |
| `band_names` length doesn't match the input's channel count | `ShapeMismatchError` |
| A non-3D input tensor | `InvalidTransformError` |

## Tests

```bash
sen2sr_venv/bin/python -m pytest \
    frame/tests/test_uncertainty_transforms.py \
    frame/tests/test_uncertainty_statistics.py \
    frame/tests/test_uncertainty_ensemble.py \
    frame/tests/test_uncertainty_report.py \
    frame/tests/test_uncertainty_geospatial_integration.py -v
```

All 69 tests use small synthetic tensors and simple deterministic fake
"models" (a plain bicubic-upsampling callable, and a deliberately
non-equivariant variant for exercising genuine positive variance) — no
network, no real `SEN2SRLite` weights required to test this package's own
logic. The geospatial integration test uses real `frame.geospatial`/
`frame.preprocessing` code with synthetic metadata, also with no network.
