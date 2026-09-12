# `frame.validation`

Phase 4 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`
Section 11.B, resolved by the Phase 3.5 validation-resource research).
Runs FRAME's proven, **unmodified** `SEN2SRLite/NonReference_RGBN_x4` path
against real LR/HR pairs from the `opensr-test` benchmark, producing three
metric groups that are reported **side by side and never merged**.

This package imports nothing from `sen2sr` and calls no model itself — the
real model call happens in `experiments/validation/run_experiment.py`,
matching the separation of concerns established in Phases 1–3
(`frame.preprocessing`, `frame.geospatial`, `frame.consistency` don't touch
`sen2sr`/`mlstac` either).

## Exact benchmark version used

| | |
|---|---|
| Package | `opensr-test` |
| Version installed | **1.3.3** (PyPI, released 2025-03-21; matches the `ESAOpenSR/opensr-test` GitHub `main` branch at commit `b42b1cba8a04b32341044f1f29474e5448499158`) |
| Dataset host | Hugging Face, `isp-uv-es/opensr-test`, repo commit `e4600b9c74a621adeec047e5f6cc7a2d70a58134` |
| Dataset format version | `v3` (`opensr_test.load(..., version="v3")`, the default) |

`pip install opensr-test==1.3.3` reproduces exactly this environment.

## The three metric groups — never merged

| Group | What it compares | Needs real HR? | Source |
|---|---|---|---|
| **(A) `standard_reference_metrics`** | bicubic baseline vs. SEN2SR, each vs. real HR | Yes | `frame.validation.reference_metrics` |
| **(B) `opensr_test_metrics`** | opensr-test's own reflectance/spectral/spatial/synthesis/hallucination/omission/improvement | Yes | `frame.validation.opensr_test_metrics`, wrapping the real `opensr_test.Metrics` class |
| **(C) `phase3_self_consistency`** | LR vs. SR only | **No** | `frame.consistency` (Phase 3, reused unchanged) |

`frame/tests/test_validation_report.py::test_report_keeps_the_three_metric_groups_separate`
enforces this structurally — `ValidationReport` has no `overall_score` or
`combined_score` field, and never will.

## Exact metric definitions and their source

### (A) Standard reference metrics

| Metric | Formula | Source |
|---|---|---|
| RMSE | `sqrt(mean((estimate − hr)²))`, masked | MSE computed via `opensr_test.distance.L2` (per-pixel mode), sqrt applied by us |
| PSNR | `10·log10(data_range² / mse)`, standard higher-is-better form | Same MSE as above; **deliberately not** `opensr_test`'s own `"psnr"` method, which computes the *reciprocal* (`IPSNR`) — see `reference_metrics.py`'s module docstring for why reusing that under the name "PSNR" would be misleading |
| SAM | `degrees(arccos(dot(x,y) / (‖x‖·‖y‖)))`, masked, per-pixel then averaged | `opensr_test.distance.SAD` (per-pixel mode) — their own internal variable is literally named `sam_score`; this *is* the standard Spectral Angle Mapper formula |
| SSIM | Standard multichannel SSIM | `skimage.metrics.structural_similarity` (already a transitive `opensr-test` dependency; `opensr-test` itself provides no SSIM) |
| ERGAS | `100·(1/scale_factor)·sqrt(mean_bands((RMSE_band / mean(HR_band))²))` | Implemented directly (Wald, 2000's standard fusion-quality formula) — neither `opensr-test` nor `scikit-image` provide it |

Each is computed **twice** per sample — once for the bicubic baseline
(`frame.validation.bicubic.bicubic_upsample`, bicubic + antialiasing,
matching the convention already used in `experiments/baseline/` and
`experiments/consistency/`), once for the real SR output — against the
identical HR reference and mask, so the bicubic-vs-SEN2SR comparison is
apples-to-apples.

### (B) opensr-test's own metrics

Computed by calling the real `opensr_test.Metrics().compute(lr, sr, hr)`
directly — not reimplemented. See `docs/Metrics/correctness.md` and
`opensr_test/main.py`/`distance.py` (read in full during Phase 4/3.5
research) for the exact upstream formulas: reflectance (L1, LR-consistency),
spectral (SAD/SAM, LR-consistency), spatial (phase-correlation misalignment,
via `satalign`), synthesis (harmonized-SR high-frequency detail vs.
bilinear-upsampled LR), and hallucination/omission/improvement (the
three-way normalized-distance classification in `correctness.md`, using
upstream's own published defaults: `correctness_temperature=0.25`,
`ha_score=om_score=im_score=0.05`, gradient mask at the 75th percentile of
LR–HR distance, `border_mask=16`). Every result records the full
`opensr_test.Config` used (`config_summary`) for reproducibility.

**Note:** `opensr_test.Metrics` has no mask parameter — group (B)'s numbers
do not respect FRAME's own nodata mask (group A and C do). Its `spatial`
value can legitimately come back `NaN` on some scenes (phase correlation
finding no coherent shift) — reported as `None`, never a bare `NaN`.

### (C) Phase 3 self-consistency

Unchanged — see `frame/consistency/README.md`. Needs only `lr` and `sr`,
reused here exactly as built in Phase 3.

## Masking

Two separate masks are used, one per grid:

- **LR-grid mask** (feeds group C) — `ValidityMask.from_nodata(lr, nodata_value=0.0)`.
- **HR-grid mask** (feeds group A) — `ValidityMask.from_nodata(hr, nodata_value=0.0)`.

Both reuse `frame.preprocessing.masks.ValidityMask` (Phase 1) rather than a
new implementation. **This is a defensive safety net, not the primary
quality-control step** — opensr-test's own README states its datasets are
"carefully crafted to minimize spatial and spectral misalignment," i.e. the
benchmark's own curation is the primary filter against cloud/misregistration
contamination; FRAME's nodata mask here only catches literal zero-fill
pixels (e.g. reprojection borders). Group (B) is not masked at all (see
above) — its own `border_mask=16` crop and gradient-threshold masking (for
correctness only) are upstream's own, different mechanism.

## Dataset structure — verified facts, not assumptions

Verified by loading the real `spot` subset during Phase 4 development
(`opensr_test.load("spot")`, 9 samples):

- A loaded subset is a plain **`dict`** (its own `load()` docstring claims
  `torch.Tensor` — that's stale/wrong) with keys `L2A`, `L1C`, `HR`,
  `HRharm`, `metadata`, plus two undocumented lowercase duplicates (`hr`,
  `hr_harm`, confirmed **not** identical to `HR`/`HRharm` by value) that
  this adapter deliberately ignores.
- `L2A`: `(N, 12, H, W)` uint16. `L1C`: `(N, 13, H, W)` uint16 (the HF
  dataset card's prose says "12 bands" for L1C — also wrong; its own band
  table, and the real array shape, both say 13). `HR`/`HRharm`:
  `(N, 4, H·scale, W·scale)` uint16.
- **L2A band order** — copied verbatim from the *"L2A Index"* column of
  [the dataset's own published band table](https://huggingface.co/datasets/isp-uv-es/opensr-test/raw/main/README.md):
  `B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12` (indices 0–11). Our
  4 target bands are therefore at L2A indices `[3, 2, 1, 7]` (B04, B03, B02,
  B08) — an authoritative, table-confirmed fact, not inferred from pixel
  correlation (which was tried during research and found too confounded by
  cross-band brightness correlation on real photographs to discriminate
  reliably on its own).
- `HR`/`HRharm`'s 4 bands are described as "RGBNIR" — read as
  `[Red, Green, Blue, NIR]` in that literal order, matching
  `frame.preprocessing.RGBN_BANDS` exactly. This is a naming-convention
  inference (weaker evidence than the L2A table above), not independently
  re-derived here.
- All four arrays are raw digital numbers scaled by 10,000 — confirmed by
  the dataset README's own literal usage example (`... / 10000`), identical
  to `frame.preprocessing.reflectance`'s convention. `extract_sample` calls
  `frame.preprocessing.to_reflectance` directly, reusing Phase 1's code.
- Per-sample `metadata` (a pandas DataFrame) carries real `crs` and `affine`
  columns (HR/SR-grid, GDAL-style 6-tuple string) — genuine georeferencing,
  usable directly with `frame.preprocessing.metadata.RasterMetadata`. The
  LR-grid transform is derived by scaling the HR transform's pixel-size
  terms by `scale_factor` (the documented inverse of
  `frame.geospatial.transform.derive_output_transform`), keeping the origin
  fixed.

## Supported subsets

Only `spot`, `spain_crops`, `spain_urban` — their LR grid is exactly
128×128 (our proven native patch size) at exactly ×4 scale.
`naip`'s LR grid is 121×121 (mismatched) and `venus` is ×2 (wrong scale for
our model) — both explicitly **out of Phase 4 scope**, not silently
accepted (`extract_sample` raises `UnsupportedSubsetError` for either).

## Storage

| Subset | Real file size (HTTP `Content-Length`, `v3`) | Samples |
|---|---|---|
| `spot` | 196,120,408 bytes ≈ 187 MB | 9 |
| `spain_crops` | 140,385,942 bytes ≈ 134 MB | 28 |
| `spain_urban` | 100,276,037 bytes ≈ 96 MB | 20 |
| **Total (our 3 primary subsets)** | **≈ 417 MB** | **57** |

Measured via `curl -I` against the exact Hugging Face resolve URLs before
any download was attempted — see `experiments/validation/README.md`. No
sample-level/streaming access exists in `opensr_test.dataset.load()` — it
downloads one whole pickle per subset — but at ~96–187 MB per subset this
is not a practical obstacle; there is no need to avoid downloading a whole
subset once its size is known this small.

## Dependencies

`opensr-test==1.3.3` was installed as a real dependency (`pip install
opensr-test`), per this phase's explicit instruction to use the official
package rather than copying its implementation. This transitively installs
`satalign`, `kornia`, `opencv-python`, `scikit-image`, `pydantic`, `mpltern`,
and (because `opensr-test`'s own `pyproject.toml` lists them as plain,
non-optional dependencies despite naming them for a `[perceptual]` extra)
`lpips`, `open-clip-torch`, and `openai-clip` — none of which
`frame.validation` calls; only `opensr_test.Metrics`, `opensr_test.distance`,
and `skimage.metrics.structural_similarity` are used. No perceptual
(LPIPS/CLIP) metric is computed anywhere in this phase.

## Scientific framing — repeated on every report

> This benchmark evaluates reconstruction against an independently sourced
> higher-resolution reference dataset. It does NOT establish that the SR
> output equals a native 2.5m Sentinel-2 observation, because no such
> native Sentinel-2 measurement exists.

`frame.validation.report.SCIENTIFIC_FRAMING` carries the full text (with the
cross-sensor confound list) and is attached verbatim to every
`ValidationReport`.

## Tests

```bash
sen2sr_venv/bin/python -m pytest \
    frame/tests/test_validation_adapter.py \
    frame/tests/test_validation_bicubic.py \
    frame/tests/test_validation_reference_metrics.py \
    frame/tests/test_validation_opensr_test_metrics.py \
    frame/tests/test_validation_report.py -v
```

All use small synthetic tensors/dicts — no network, no real dataset
download. (`opensr_test.Metrics` itself makes no network calls — only
`opensr_test.load`/`frame.validation.load_subset` do — so groups (B)'s
tests exercise the *real* `opensr_test.Metrics` class, not a mock.)
