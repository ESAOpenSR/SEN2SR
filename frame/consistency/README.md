# `frame.consistency`

Phase 3 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`,
Section 9.2). Independent, **post-hoc** diagnostics that measure how well an
SR output remains consistent with the original Sentinel-2 observation it was
derived from.

This package imports nothing from `sen2sr` and calls no model. It only
consumes whatever tensors `frame.preprocessing` prepared and the (unmodified)
upstream model produced. See `experiments/consistency/` for the reference
integration against a real model run.

## This is a self-consistency check, not an accuracy metric

Every number this package produces answers one question: **"does the SR
output's coarse-scale content still agree with the real sensor measurement
it was derived from?"** It never answers **"is the SR output accurate?"** —
that would require an independent higher-resolution reference image, which
does not exist for Sentinel-2 (`docs/FRAME_TECHNICAL_SPEC.md` Section 11).
A discrepancy here can indicate a genuine problem (a masking bug, a
band-order mistake, a numerically degenerate tile) — but a *low* discrepancy
only proves the output is self-consistent with its own input, never that it
is correct.

## Relationship to the existing upstream Fourier hard constraint

`sen2sr/models/tricks.py`'s `FourierHardConstraint`/`HardConstraint`
(**unmodified**, still runs on every inference inside `nonreference.py`,
`referencex2.py`, `referencex4.py`) is a **structural** mechanism: it
bicubic-upsamples the LR input to the SR size, then recombines the SR
output's high frequencies with the upsampled LR's low frequencies in the
Fourier domain — this happens *during* inference and changes what the model
is allowed to output.

`frame.consistency` is a **diagnostic** mechanism: it runs *after*
inference, on the finished SR output, and produces a number describing how
well that constraint actually held on this particular result. It does not
re-implement the constraint, does not assume it always works, and cannot
change the SR output in any way — see `frame.consistency.band_discrepancy`'s
module docstring for the precise relationship, and note the two mechanisms
deliberately use *different* resampling operations (bicubic-with-antialiasing
upsampling vs. parameter-free area-average downsampling) for different
purposes — see `frame.consistency.downsample`.

## Exact mathematical definitions

### 1. Downsample-consistency (`frame.consistency.band_discrepancy`)

Given an LR tensor `X` (bands, H, W) and its SR output `Y` (bands, H·s, W·s)
at scale factor `s`:

1. **Downsample.** `Ŷ = downsample_to_lr_grid(Y, s)` — non-overlapping
   area-average pooling (`torch.nn.functional.avg_pool2d`, kernel = stride =
   `s`): each output pixel is the arithmetic mean of the `s × s` block of SR
   pixels it corresponds to. See `frame.consistency.downsample`'s module
   docstring for why this specific, parameter-free method was chosen.
2. **Error.** `E = Ŷ − X`, computed only at valid pixels (see Masking below).
3. **Per-band statistics**, over that band's valid pixels:
   - `mean_abs_error = mean(|E|)`
   - `rmse = sqrt(mean(E²))`
   - `max_abs_error = max(|E|)`
   - `normalized_rmse = rmse / mean(|X|)` when `mean(|X|) > 0`, else not
     computed (`None`) — this is a plain normalization, not a threshold.
4. **Overall statistics** pool every band's valid samples together before
   computing the same four quantities. `valid_pixel_count` (spatial, i.e.
   how many `(row, col)` locations were valid) is reported once and is
   identical across the overall result and every per-band result, since the
   mask is spatial and shared across bands; `sample_count` (how many scalar
   error values actually went into that specific statistic —
   `valid_pixel_count` for one band, `valid_pixel_count × num_bands` for the
   pooled overall result) is recorded separately so the exact calculation
   is reproducible from the report alone.

### 2. Band-wise spectral shape check (`frame.consistency.spectral_ratios`)

Computed from `X` and `Ŷ` (the same downsampled SR array as above, reused,
never recomputed):

- **NDVI** = `(NIR − Red) / (NIR + Red)` = `(B08 − B04) / (B08 + B04)`.
- **B08/B04 ratio** = `B08 / B04`.

Each index is computed once from `X` and once from `Ŷ`; the reported
statistic is `mean(|index(Ŷ) − index(X)|)` (plus `rmse` and `max_abs`) over
pixels where the index was computable in *both*. A pixel whose denominator
(`|NIR + Red|` for NDVI, `|B04|` for the ratio) has magnitude at or below
`DIVISION_EPSILON = 1e-6` is excluded as **not computable at that pixel** —
never divided into a fabricated near-infinite value. `DIVISION_EPSILON` is a
numerical floor to keep division well-defined, not a scientific threshold.

### 3. Cross-tile overlap (`frame.consistency.tiles`)

Given two SR tile arrays placed on a shared pixel grid (by their declared
`(row, col)` offsets), the pixel-space rectangle where they overlap is
computed, masked pixels from *either* tile are excluded, and the same
`mean_abs_discrepancy` / `rmse` / `max_abs_discrepancy` statistics are
computed over the raw pixel-value difference in that region.

**Read `frame/consistency/tiles.py`'s module docstring before using this
for a real multi-tile run.** `sen2sr.utils.predict_large` does not expose
per-tile SR outputs before they are cropped and blended into its single
returned tensor — verified by reading `sen2sr/utils.py` in full. This
function is a ready-to-use, pure utility against an *equivalent
abstraction* (two tile arrays + their pixel offsets), for a future
FRAME-owned tiling orchestrator that does not yet exist. It is exercised
in this project only against synthetic tiles (`frame/tests/test_consistency_tiles.py`);
Baseline 0's real scene is a single 128×128 patch and never triggers
`predict_large`'s multi-tile path, so `experiments/consistency/` reports
`cross_tile = None` by construction, not a fabricated result — see that
experiment's README for the explicit statement of this limitation.

## Masking

Every diagnostic above takes a boolean validity mask
(`frame.preprocessing.ValidityMask.array`, shape `(H, W)`) and excludes
`False` (masked/cloud/nodata) pixels *before* computing any statistic. A
masked pixel is never treated as a real zero-reflectance observation — this
is the same guarantee `frame.preprocessing` itself provides, carried through
into every diagnostic here.

## Threshold policy — read before adding one

**No numeric pass/fail threshold exists anywhere in this package.** A
threshold ("discrepancy must be below X to be acceptable") requires an
empirical discrepancy distribution from many real runs to be meaningful —
this project has run the diagnostic on exactly one deterministic scene so
far (`experiments/consistency/`), which is nowhere near enough evidence to
set one. Every result type here (`BandDiscrepancy`, `DownsampleConsistencyResult`,
`SpectralIndexComparison`, `TileOverlapResult`, `ConsistencyDiagnostics`)
exposes only:

- raw diagnostic values (`mean_abs_error`, `rmse`, `max_abs_error`,
  `normalized_rmse`, `mean_abs_discrepancy`, ...), each `None` — not `0.0`,
  not `NaN` — when there was nothing valid to compute it from, and
- a `frame.consistency.status.ComputationStatus`: `COMPUTABLE`,
  `NOT_COMPUTABLE`, or `INVALID_INPUT` — which states whether a number
  *could be computed*, never whether it is *good or bad*.

`frame/tests/test_consistency_report.py::test_no_pass_fail_threshold_fields_on_result_types`
enforces this structurally (no result dataclass may carry a field named
anything like `passed`/`is_valid`/`threshold`/`accept`/`reject`/`good`/`bad`).

## Public API

```python
from frame.consistency import (
    run_consistency_diagnostics,     # top-level: everything at once
    ConsistencyDiagnostics,
    compute_downsample_consistency,  # downsample-consistency only
    compute_ndvi_comparison,         # NDVI shape check only
    compute_b08_b04_ratio_comparison,# B08/B04 shape check only
    compare_tile_overlap,            # cross-tile, given tile arrays + offsets
    ComputationStatus,
)

diagnostics = run_consistency_diagnostics(
    lr,            # torch.Tensor (bands, H, W) -- the real LR observation
    sr,            # torch.Tensor (bands, H*scale, W*scale) -- the model output
    mask.array,    # np.ndarray (H, W) bool -- frame.preprocessing.ValidityMask
    band_names=("B04", "B03", "B02", "B08"),
    scale_factor=4,
)

print(diagnostics.downsample_consistency.overall.rmse)
print(diagnostics.ndvi_comparison.status)
```

Raises a specific `ConsistencyError` subclass for structurally invalid input
(shape mismatch, wrong scale factor, mask shape mismatch, missing band):

| Problem | Exception |
|---|---|
| LR/SR not 3-D, or band counts disagree | `ShapeMismatchError` |
| SR spatial shape ≠ LR shape × scale_factor | `ShapeMismatchError` |
| `scale_factor` not a positive integer | `ScaleFactorError` |
| Mask shape doesn't match the LR grid | `InvalidMaskError` |
| A spectral index needs a band not present in `band_names` | `MissingBandError` |

## Tests

```bash
sen2sr_venv/bin/python -m pytest \
    frame/tests/test_consistency_downsample.py \
    frame/tests/test_consistency_band_discrepancy.py \
    frame/tests/test_consistency_spectral_ratios.py \
    frame/tests/test_consistency_tiles.py \
    frame/tests/test_consistency_report.py -v
```

All 57 tests use small synthetic arrays only — no network, no model
weights, no `sen2sr`/`mlstac`/`cubo` imports required.

## Limitations

- Downsample-consistency and the spectral-shape checks measure agreement
  with the *low-frequency, coarse-scale* content of the LR observation —
  by design, since that is exactly what the upstream hard constraint
  targets. They say nothing about whether the SR output's *invented*
  high-frequency detail is correct, because there is nothing to compare
  that detail against (Section 11.D).
- The area-average downsampling method is an approximation of "what a 10 m
  sensor pixel would have measured" — it assumes uniform spatial response
  within a pixel, which is a simplification of real sensor optics, not a
  claim of physical equivalence.
- Cross-tile comparison is implemented and unit-tested but not yet
  exercised against a real multi-tile `sen2sr` run (see above).
- These diagnostics run on one image at a time; there is no aggregation
  across many runs yet (that would be part of a future validation harness,
  `docs/FRAME_TECHNICAL_SPEC.md` Section 11, explicitly out of scope for
  Phase 3).
