# `frame.geospatial`

Phase 2 of the FRAME implementation plan (`docs/FRAME_TECHNICAL_SPEC.md`,
Section 10). Preserves and reconstructs geospatial metadata (CRS, affine
transform, bounds) around the tensor-based, **unmodified** upstream
`sen2sr` SR pipeline, and writes/reads the SR result as a real GeoTIFF.

This package imports nothing from `sen2sr` and calls no model. It only
handles geospatial context and file I/O around whatever tensor the model
produced. See `experiments/baseline_geoexport/` for the reference
integration with a real model run.

## Scope of this phase

Supported today: deterministic output-transform derivation and GeoTIFF
export/import for the proven 4× RGBN path (`RGBN_SCALE_FACTOR = 4`,
10 m → 2.5 m), with a general (non-4×-specific, rotation/shear-aware)
implementation underneath.

Not in this phase: reprojection (this layer **never** reprojects input
data — a CRS mismatch or a missing CRS is a hard error, not something this
code silently fixes), tiling orchestration across multiple GeoTIFF tiles,
uncertainty rasters, validation metrics, downstream applications, the API,
and the frontend.

## Why the raster/metadata representation is reused, not duplicated

`frame.preprocessing.metadata.RasterMetadata` (Phase 1) already carries
every field this phase's contract asks for: CRS, affine transform, bounds,
resolution, width, height, band names, and nodata value — plus the
provenance fields (`acquisition_timestamp`, `sr_variant`,
`cloud_mask_coverage`) this phase writes as GeoTIFF tags. Defining a second,
overlapping dataclass here would create two competing sources of truth for
the same raster's geospatial identity, so `frame.geospatial.metadata`
re-exports `RasterMetadata` and adds the operations this layer needs on top
of it.

## Why `rasterio`

Writing a spec-compliant GeoTIFF (embedded CRS, geotransform, per-band
descriptions, nodata, custom tags) and reading one back for independent
verification requires an actual raster I/O library — reimplementing
GeoTIFF's tag/IFD structure by hand would mean reinventing GDAL, which
`rasterio` already wraps correctly and is exactly what it's for. The
dependency is confined to `frame/geospatial/geotiff.py`; every other module
in this package (`metadata.py`, `transform.py`) has no `rasterio` import
and operates on plain tuples/floats. `rasterio` was already present as a
transitive dependency of `cubo`/`stackstac` in the Phase 1 environment, so
no new package needed to be installed — see the root README/Phase 1 report
for the exact dependency list.

## Public API

```python
from frame.geospatial import (
    RasterMetadata,             # reused from frame.preprocessing.metadata
    derive_output_metadata,      # deterministic output-transform derivation
    RGBN_SCALE_FACTOR,           # = 4, the proven RGBN path's scale factor
    write_geotiff, read_geotiff,
)

# after running preprocess_rgbn(...) and the model:
output_metadata = derive_output_metadata(
    preprocessed.metadata,             # frame.preprocessing.PreprocessedInput.metadata
    scale_factor=RGBN_SCALE_FACTOR,
    output_band_names=preprocessed.metadata.band_names,  # SR doesn't change bands
)

write_geotiff("sr_output.tif", sr_tensor.numpy(), output_metadata)

array, read_back_metadata = read_geotiff("sr_output.tif")
```

`write_geotiff`/`derive_output_metadata` raise a specific `GeospatialError`
subclass rather than proceeding with an invented value:

| Problem | Exception |
|---|---|
| CRS missing on input/output metadata | `MissingCRSError` |
| Affine transform missing | `MissingTransformError` |
| Recomputed output footprint disagrees with the input's declared bounds | `InconsistentMetadataError` |

## The geospatial contract

### 1. Output-transform derivation (`frame.geospatial.transform`)

For a GDAL-style affine 6-tuple `(a, b, c, d, e, f)` and an SR
`scale_factor`:

- **Same CRS** — carried through unchanged; this layer never reprojects.
- **Pixel size divided by the scale factor** — `a`, `d`, `e` (and `b`, the
  full linear part of the transform) all divide by `scale_factor`, so
  rotation/shear terms scale consistently rather than being dropped or
  zeroed.
- **Origin preserved** — `c`, `f` (the world coordinate of the upper-left
  pixel corner) are copied unchanged.
- **Footprint (bounds) preserved** — not just assumed: `derive_output_metadata`
  recomputes the output footprint from the derived transform and the
  scaled-up width/height (via `bounds_from_transform`, which maps all four
  pixel-grid corners through the transform — correct even under rotation),
  and raises `InconsistentMetadataError` if that recomputed footprint
  drifts from the input's declared bounds by more than a floating-point
  tolerance. This is a self-consistency check, not a scientific threshold.
- **Width/height scale up by `scale_factor`.**
- **Band names** are passed explicitly (`output_band_names`) rather than
  assumed, since SR here doesn't change which bands exist, only pixel
  density — the caller states this rather than the function guessing.

### 2. GeoTIFF writing (`frame.geospatial.geotiff.write_geotiff`)

- Float32 reflectance values, written directly (no rescaling to integer
  digital numbers in this phase).
- Band order in the file == `metadata.band_names` order, with each band's
  description tag set to its name.
- CRS, affine transform, and nodata embedded via `rasterio`.
- Provenance carried as custom GeoTIFF tags: `FRAME_SR_VARIANT`,
  `FRAME_SOURCE_ACQUISITION_TIMESTAMP`, `FRAME_CLOUD_MASK_COVERAGE`,
  `FRAME_PIPELINE`.
- Refuses to write (raises `MissingCRSError`/`MissingTransformError`) if
  the metadata doesn't have a real CRS/transform — **never fabricates
  one**.

### 3. GeoTIFF reading (`frame.geospatial.geotiff.read_geotiff`)

- Reads the array plus a `RasterMetadata` reconstructed from the file's own
  embedded CRS/transform/bounds/band descriptions/tags — for independent
  verification that what was written is what comes back.
- `require_crs=True` by default: raises `MissingCRSError` if the opened
  file has no CRS, rather than silently returning `crs=None`. Pass
  `require_crs=False` to inspect a file that may legitimately lack one.

## Tests

```bash
sen2sr_venv/bin/python -m pytest frame/tests/test_geospatial_metadata.py \
    frame/tests/test_geospatial_transform.py frame/tests/test_geospatial_geotiff.py -v
```

All tests use small synthetic rasters (`numpy` arrays + `pytest`'s
`tmp_path`). No network access, no Hugging Face weights, no
`sen2sr`/`mlstac`/`cubo` imports required.
