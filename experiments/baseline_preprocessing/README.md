# Baseline + Preprocessing — Phase 1 equivalence experiment

Proves that `frame.preprocessing.preprocess_rgbn` (the new FRAME preprocessing
layer, see `frame/preprocessing/README.md`) is a **faithful, numerically
equivalent, more robust replacement** for the ad hoc inline preprocessing in
`experiments/baseline/run_baseline.py` (Baseline 0) — not an improvement in
SR quality, and not a new benchmark.

This experiment reads Baseline 0's own saved output tensors
(`experiments/baseline/outputs/input_tensor.pt`, `sr_tensor.pt`) as its
reference. **It does not modify `experiments/baseline/` in any way.**

## What this proves

1. Given the *same* raw fetched Sentinel-2 scene, `preprocess_rgbn` produces
   a model-input tensor numerically equivalent to Baseline 0's proven input
   tensor — both when the raw bands already arrive in canonical
   `(B04, B03, B02, B08)` order, and when they are deliberately scrambled
   first (exercising the reorder-bands logic against real data, not just
   the synthetic arrays used in `frame/tests/`).
2. Feeding that preprocessed tensor into the *same unmodified* upstream
   `SEN2SRLite/NonReference_RGBN_x4` model that Baseline 0 uses produces an
   SR output numerically equivalent to Baseline 0's saved SR output.

## What this does NOT prove

Nothing about SR accuracy or quality. There is still no reference
high-resolution image anywhere in this experiment — see
`docs/FRAME_TECHNICAL_SPEC.md` Section 11 for why that validation question
is separate and deliberately deferred. This experiment is scoped
exclusively to preprocessing-layer correctness.

## Result of the last run

All four comparisons came back **bit-exact** (`max_abs_diff = 0.0`,
well within the documented tolerances below) on this machine/GPU:

| Comparison | atol | max abs diff | Result |
|---|---|---|---|
| our in-order input vs. Baseline 0's saved input | 1e-5 | 0.0 | equivalent |
| our scrambled-then-reordered input vs. Baseline 0's saved input | 1e-5 | 0.0 | equivalent |
| our in-order input vs. our scrambled-then-reordered input | 1e-5 | 0.0 | equivalent |
| our SR output vs. Baseline 0's saved SR output | 1e-4 | 0.0 | equivalent |

The full numbers for the most recent run are in
`metadata/run_metadata.json`; re-running regenerates them (see below).

### Why the tolerances are what they are

`INPUT_TENSOR_ATOL = 1e-5` and `SR_OUTPUT_ATOL = 1e-4` in
`run_experiment.py` are **floating-point comparison tolerances**, not a
scientific accuracy claim — they only bound acceptable numerical noise
between two code paths computing (in the input-tensor case) the literal
same arithmetic, and (in the SR-output case) a full model forward pass that
could in principle differ slightly by GPU kernel/reduction order even with
identical input. Per `docs/FRAME_TECHNICAL_SPEC.md`'s instruction not to
invent scientific thresholds, no attempt is made here to claim these
tolerances mean anything about model accuracy — they exist solely to
declare "close enough to call the same" for a determinism/equivalence
check.

## How the comparison is structured

1. Fetch the **exact same deterministic scene** Baseline 0 uses: same AOI
   (lat/lon), same single-day date window (`2023-01-15`–`2023-01-16`), same
   time index (`0`), same bands, same 128 px edge size, same 10 m
   resolution. These constants are duplicated from
   `experiments/baseline/run_baseline.py` (not imported, to keep both
   experiments standalone) — if that file's scene definition ever changes,
   this file's constants must be updated to match, and a runtime assertion
   at import time checks `frame.preprocessing.RGBN_BANDS` still matches the
   band list used here.
2. Reuse Baseline 0's already-downloaded model weights cache
   (`~/.cache/sen2sr_baseline/SEN2SRLite_RGBN` by default, override with
   `SEN2SR_BASELINE_WEIGHTS_DIR`) — no re-download.
3. Derive geospatial metadata (CRS, affine transform, bounds) from the
   fetched `cubo` cube. `cubo.create()` deliberately deletes the
   `crs`/`transform` attributes `stackstac` would otherwise set (see
   `cubo/cubo.py`), keeping only `epsg` and `resolution` — so the transform
   and bounds are recomputed here from those plus the cube's own
   pixel-center `x`/`y` coordinate arrays, using standard north-up raster
   geometry. This is exactly the kind of metadata capture
   `docs/FRAME_TECHNICAL_SPEC.md` Section 7 calls for, and it is passed to
   `preprocess_rgbn(..., require_geospatial=True)` to prove the metadata
   path works end-to-end against a real source, not just a synthetic one.
4. Run `preprocess_rgbn` twice on the one fetched raw array: once with
   bands already in canonical order, once with the same raw array's band
   axis and band-name list deliberately permuted beforehand — so any
   difference between the two results is attributable only to the
   reordering logic, not to a second network fetch.
5. Compare both resulting tensors against Baseline 0's saved
   `input_tensor.pt`.
6. Run the real, unmodified upstream model on our preprocessed tensor and
   compare the result against Baseline 0's saved `sr_tensor.pt`.
7. Save a numeric comparison report (`metadata/run_metadata.json`), the
   preprocessed input/output tensors, and a visual diff map
   (`outputs/sr_diff_vs_baseline0.png` — expected to render as visually
   flat/uniform, since the underlying difference is zero).

## Layout

```
experiments/baseline_preprocessing/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/              # tensors + diff visualization (created by the script)
  metadata/             # run_metadata.json (created by the script)
```

## Running it

Uses the same isolated environment as Baseline 0
(`sen2sr_venv/`, Python 3.11, created via `uv venv --python 3.11
sen2sr_venv`), with the same dependencies Baseline 0 documents
(`torch`, `torchvision`, `timm`, `einops`, `tqdm`, `sen2sr`, `mlstac`,
`cubo`, `matplotlib`) plus `numpy` and `pytest` for the `frame` package
itself. No new dependency was added beyond `pytest` (dev-only, for
`frame/tests/`) — see `frame/preprocessing/README.md` and this repo's top
level for the exact install commands used.

From the repository root:

```bash
sen2sr_venv/bin/python experiments/baseline_preprocessing/run_experiment.py
```

Requires internet access (a live STAC endpoint for the imagery; the model
weights are expected to already be cached from running Baseline 0 first —
if not, they will be downloaded automatically). Requires
`experiments/baseline/outputs/{input_tensor.pt,sr_tensor.pt}` to already
exist (i.e., run `experiments/baseline/run_baseline.py` at least once
first).

The script exits with a non-zero status if any comparison falls outside
its documented tolerance, so it can be used as a regression check that a
future change to `frame/preprocessing/` has not silently altered the
proven inference behavior.
