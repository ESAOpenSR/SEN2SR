# Baseline + geospatial export — Phase 2 experiment

Proves that `frame.geospatial` (see `frame/geospatial/README.md`) correctly
preserves and reconstructs geospatial context around the tensor-based,
**unmodified** upstream `sen2sr` SR pipeline, end to end against a real
scene: fetch → FRAME preprocessing → unmodified model → geospatial export
→ independent reopen and verification.

This experiment reuses Baseline 0's exact deterministic scene definition
(same AOI, same date window, same time index, same bands, same model). It
does **not** modify `experiments/baseline/` or
`experiments/baseline_preprocessing/` in any way.

## What this proves

1. Geospatial metadata (CRS, affine transform, bounds) captured from the
   real fetched scene *before* any tensor conversion survives
   `frame.preprocessing.preprocess_rgbn` unchanged.
2. `frame.geospatial.derive_output_metadata` produces an output transform
   whose recomputed footprint exactly matches the input footprint, and
   whose pixel size is exactly the input pixel size divided by the model's
   scale factor (4×, 10 m → 2.5 m, for this path).
3. The GeoTIFF written from the real SR tensor (`frame.geospatial.write_geotiff`),
   when reopened **independently** (`frame.geospatial.read_geotiff`,
   a fresh `rasterio.open()` call on the file on disk — not the in-memory
   objects used to write it), reports the same CRS, transform, bounds,
   dimensions, resolution, and band order that were intended, and the same
   pixel values that were written.

## What this does NOT prove

Nothing about SR accuracy or quality — see
`docs/FRAME_TECHNICAL_SPEC.md` Section 11. This experiment is scoped
exclusively to geospatial-layer correctness.

## Result of the last run

All checks passed on this machine/GPU:

| Check | Result |
|---|---|
| CRS matches | OK — `EPSG:32630` preserved end to end |
| Transform matches | OK |
| Bounds match | OK |
| Width/height match | OK — `128×128` → `512×512` |
| Resolution matches | OK — `10.0 m` → `2.5 m` |
| Band order matches | OK — `(B04, B03, B02, B08)` |
| Pixel values match (in-memory tensor vs. reopened file) | OK |
| **Invariant: input footprint == output footprint** | OK — drift `0.0 m` |
| **Invariant: output pixel size == input pixel size / 4** | OK — `2.5 m == 10.0 / 4` |

Independent inspection of the written file (a fresh `rasterio.open()` call,
separate from the experiment's own verification code) confirms: driver
`GTiff`, CRS `EPSG:32630`, transform `(2.5, 0, 720285, 0, -2.5, 4375125)`,
bounds `(720285, 4373845, 721565, 4375125)`, size `512×512`, 4 float32
bands with descriptions `('B04', 'B03', 'B02', 'B08')`, nodata `0.0`, and
custom tags `FRAME_SR_VARIANT`, `FRAME_SOURCE_ACQUISITION_TIMESTAMP`,
`FRAME_CLOUD_MASK_COVERAGE`, `FRAME_PIPELINE`.

The full numbers for the most recent run are in
`metadata/run_metadata.json`; re-running regenerates them.

## How the experiment is structured

1. Fetch the **exact same deterministic scene** Baseline 0 uses (same
   constants, duplicated rather than imported — see
   `experiments/baseline_preprocessing/README.md` for why each experiment
   stays standalone).
2. Reuse Baseline 0's already-downloaded model weights cache — no
   re-download.
3. Extract geospatial metadata (CRS, transform, bounds) from the fetched
   `cubo` cube **before** converting anything to a tensor, using the same
   coordinate-arithmetic approach as
   `experiments/baseline_preprocessing/run_experiment.py` (`cubo` deletes
   `stackstac`'s own crs/transform attrs; they're recomputed here from
   `epsg` + the cube's pixel-center `x`/`y` coordinates).
4. Run `preprocess_rgbn(..., require_geospatial=True)` — geospatial context
   is mandatory here, not optional, since the whole point of this
   experiment is proving it survives the pipeline.
5. Run the real, unmodified upstream model.
6. `derive_output_metadata(preprocessed.metadata, scale_factor=RGBN_SCALE_FACTOR, ...)`
   then `write_geotiff(...)`.
7. `read_geotiff(...)` on the just-written file, and compare every field
   against what was intended.
8. Explicitly assert the two invariants named in the Phase 2 requirements
   (footprint preservation; pixel size == input / 4), each printed and
   recorded independently of the general field-by-field check in step 7.
9. Save `metadata/run_metadata.json`, the GeoTIFF itself
   (`outputs/sr_output.tif`), and an RGB preview rendered from the
   **reopened** file (`outputs/sr_preview_rgb.png`) — so the preview is
   itself evidence the round trip through disk worked, not just a preview
   of the in-memory tensor.

## Layout

```
experiments/baseline_geoexport/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/              # sr_output.tif, sr_preview_rgb.png
  metadata/              # run_metadata.json
```

## Running it

Same environment as Baseline 0 / Phase 1
(`sen2sr_venv/`). No new dependency was added for this phase beyond what
Phase 1 already required — `rasterio` (used only inside
`frame/geospatial/geotiff.py`) was already present as a transitive
dependency of `cubo`/`stackstac`.

From the repository root:

```bash
sen2sr_venv/bin/python experiments/baseline_geoexport/run_experiment.py
```

Requires internet access (a live STAC endpoint) and the same model weights
cache Baseline 0 downloads (auto-downloaded if not already cached). Exits
with a non-zero status if any verification check or invariant fails, so it
can be used as a regression check for `frame/geospatial/`.
