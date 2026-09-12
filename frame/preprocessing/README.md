# `frame.preprocessing`

Phase 1 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`, Section 7).

Converts a raw Sentinel-2 band stack into a validated, model-ready
representation for the existing, **unmodified** upstream `sen2sr`
`SEN2SRLite/NonReference_RGBN_x4` inference path — the same path exercised
by `experiments/baseline/run_baseline.py` (Baseline 0).

This package imports nothing from `sen2sr` and calls no model. It only
prepares the model's input. Running the model itself is done by the caller
(see `experiments/baseline_preprocessing/` for the reference integration).

## Scope of this phase

Supported today: the proven RGBN path only —
bands `B04, B03, B02, B08` at 10 m native resolution, reflectance-normalized
the same way `experiments/baseline/run_baseline.py` does it.

Not in this phase (see `docs/FRAME_TECHNICAL_SPEC.md` Sections 19/22 for
where these land): the full 10-band reference cascade, geospatial tiling
orchestration (`sen2sr.utils.predict_large` is reused as-is, unmodified, in
a later phase), uncertainty estimation, validation metrics, downstream
applications, the API, and the frontend.

## Public API

```python
from frame.preprocessing import preprocess_rgbn, PreprocessedInput, RGBN_BANDS

result: PreprocessedInput = preprocess_rgbn(
    array,                          # np.ndarray, shape (4, H, W), any band order
    band_names=["B08", "B02", "B04", "B03"],  # what `array`'s axis 0 actually is
    input_scale="raw_digital_number",          # or "reflectance" -- never guessed
    resolution_m=10.0,
    nodata_value=0.0,               # optional
    scl=scl_array,                  # optional Sentinel-2 L2A SCL band, shape (H, W)
    crs="EPSG:32630",                # optional geospatial context to preserve
    transform=(10.0, 0.0, 500000.0, 0.0, -10.0, 4400000.0),
    bounds=(500000.0, 4398720.0, 501280.0, 4400000.0),
    acquisition_timestamp="2023-01-15T10:54:11.024000",
    require_geospatial=False,        # True to hard-require crs/transform
)

result.tensor     # torch.Tensor, float32, shape (4, H, W), band order == RGBN_BANDS
result.mask       # ValidityMask -- separate from the tensor, never confused with real zeros
result.metadata   # RasterMetadata -- provenance/geospatial record
```

`preprocess_rgbn` raises a specific `PreprocessingError` subclass (see
`frame.preprocessing.errors`) for every unsupported input, rather than
coercing or silently proceeding:

| Problem | Exception |
|---|---|
| Wrong shape / not square / wrong patch size | `InvalidShapeError` |
| Missing/extra/duplicate band | `UnsupportedBandsError` |
| Resolution != 10 m | `UnsupportedResolutionError` |
| Unknown `input_scale` value | `InvalidInputScaleError` |
| `require_geospatial=True` but no CRS/transform given | `MissingMetadataError` |

## The preprocessing contract

1. **Input validation** (`frame.preprocessing.validation`) — shape must be a
   square `(4, H, W)` stack (128×128 by default, matching the model's
   proven patch size used in Baseline 0; pass `patch_size=None` to skip this
   check for a future tiling-orchestration phase, or a different int for a
   different fixed size). Band set must exactly match `RGBN_BANDS`
   (`B04, B03, B02, B08`) — any order accepted as input, always reordered to
   that canonical order in the output. Resolution must equal 10.0 m within
   floating-point tolerance.

2. **Reflectance preprocessing** (`frame.preprocessing.reflectance`) —
   reproduces the proven Baseline 0 / upstream README arithmetic exactly:
   `(digital_number / 10_000).astype("float32")` when
   `input_scale="raw_digital_number"`, then
   `nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)`. When
   `input_scale="reflectance"`, the `/10_000` step is skipped — **which
   scale the input is in is always stated explicitly by the caller, never
   guessed from the data**, so an already-normalized input is never
   silently divided by 10,000 a second time.

3. **Validity/mask handling** (`frame.preprocessing.masks`) — a
   `ValidityMask` is tracked *separately* from the reflectance tensor:
   - `ValidityMask.from_nodata(array, nodata_value)`: a pixel is invalid iff
     *every* band reads `nodata_value` there (a pixel that's incidentally
     zero in only one band is still a real observation).
   - `ValidityMask.from_scl(scl, invalid_classes=None)`: flags Sentinel-2
     L2A Scene Classification codes as invalid. Defaults to ESA's own SCL
     codes for no-data/saturated/cloud-shadow/cloud/cirrus
     (`{0, 1, 3, 8, 9, 10}` — `DEFAULT_SCL_INVALID_CLASSES`), overridable.
   - Masks combine via `.combine()` (logical AND: valid only where every
     mask agrees). `.coverage()` returns the fraction of *valid* pixels.
   - A masked pixel's reflectance value is not touched or specially set —
     it goes through the same NaN/Inf cleanup as any other pixel — so
     **the mask, not the tensor value, is the only authoritative record of
     validity**. Never infer validity from whether a reflectance value
     happens to be `0.0`.

4. **Metadata** (`frame.preprocessing.metadata.RasterMetadata`) — an
   immutable record with: `crs`, `transform` (GDAL-style affine 6-tuple,
   `None` if unknown), `bounds`, `resolution_m`, `width`, `height`,
   `band_names`, `acquisition_timestamp`, `nodata_value`,
   `cloud_mask_coverage`, `sr_variant`. Use `RasterMetadata.unknown(...)`
   when no geospatial context exists (e.g. a synthetic test array).
   `require_geospatial()` raises `MissingMetadataError` unless both `crs`
   and `transform` are present — called automatically by `preprocess_rgbn`
   when `require_geospatial=True`. This package does **not** depend on
   `rasterio`/`affine`/`pyproj`: it only preserves whatever geospatial
   values the caller supplies; interpreting/reprojecting them is a later
   phase (`docs/FRAME_TECHNICAL_SPEC.md` Section 10).

5. **Model input preparation** — the returned `tensor` is exactly what
   Baseline 0 constructs before calling
   `model(X[None]).squeeze(0)`: a `torch.float32` tensor, shape `(4, H, W)`,
   band order `(B04, B03, B02, B08)`. This package does not call the model
   and does not reimplement `sen2sr.utils.predict_large` tiling.

## Tests

```bash
sen2sr_venv/bin/python -m pytest frame/tests/ -v
```

All tests use small synthetic in-memory arrays. No network access, no
Hugging Face weights, no `sen2sr`/`mlstac`/`cubo` imports required.
