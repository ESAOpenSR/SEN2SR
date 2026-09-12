# FRAME API (Phase 7)

A FastAPI backend that exposes the tested FRAME pipeline (Phases 1-6) over
HTTP for the SIH prototype/demo. It is **not** a production distributed
job system: execution is synchronous, job state lives in an in-memory
registry for the life of the process, and there is no queue, worker pool,
or database. This is deliberate — see [Prototype behavior](#prototype-behavior-read-this-first).

This package only orchestrates the existing, already-tested `frame.*`
modules (`frame.preprocessing`, `frame.geospatial`, `frame.uncertainty`,
`frame.consistency`, `frame.analysis`). It never reimplements their logic
and never modifies `sen2sr/`.

## Running it

```bash
sen2sr_venv/bin/uvicorn frame.api.app:app --reload --port 8000
```

Interactive docs (Swagger UI) are then at `http://127.0.0.1:8000/docs`.

## Configuration

Every setting has a default and can be overridden by an environment
variable — nothing is hard-coded to a developer machine.

| Variable | Default | Meaning |
|---|---|---|
| `FRAME_API_WORKSPACE_DIR` | `~/.cache/frame_api/workspace` | Where uploaded rasters and generated job artifacts (GeoTIFFs, tensors) are written. Never inside a source-code directory. |
| `SEN2SR_BASELINE_WEIGHTS_DIR` | `~/.cache/sen2sr_baseline/SEN2SRLite_RGBN` | Model weights cache — reuses the *same* variable every prior phase's experiment scripts already use, so an already-downloaded cache is picked up with no re-download. |
| `FRAME_API_DEVICE` | `auto` | `auto` resolves to `cuda` if available, else `cpu`. Set to `cpu`/`cuda` to force. |
| `FRAME_API_UNCERTAINTY_SEED` | `42` | Default TTA ensemble seed for `/sr/run` when the request doesn't override it. |
| `FRAME_API_CORS_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed CORS origins for a local frontend dev server. Explicit, never `*`. |

## Endpoints

### `GET /health`
Liveness/version check. Returns `status`, `api_version`, `frame_version`
(the installed `frame` package's `__version__` if it defines one, else
`null`), and `model_name`.

### `POST /aoi/preview`
Validates an AOI/date-window request and reports the shapes and
resolutions a run would produce. **Does not fetch imagery** — Phase 7
does not add a Sentinel-2 STAC-fetch pipeline; the AOI-fetch stage was
implemented only inside `experiments/`, not the API, in prior phases. Use
this to let a frontend sanity-check a request before asking the user to
supply a real GeoTIFF via `/upload`.

Request:
```json
{"lat": 39.49, "lon": -0.43, "start_date": "2023-01-15", "end_date": "2023-01-16", "edge_size": 128}
```

Response (200):
```json
{
  "valid": true,
  "coordinates": {"lat": 39.49, "lon": -0.43},
  "date_window": {"start": "2023-01-15", "end": "2023-01-16"},
  "edge_size": 128,
  "bands": ["B04", "B03", "B02", "B08"],
  "expected_input_shape": [4, 128, 128],
  "expected_output_shape": [4, 512, 512],
  "native_resolution_m": 10.0,
  "sr_resolution_m": 2.5,
  "sr_product_description": "SR-derived product — 2.5 m pixel grid",
  "note": "..."
}
```
`start_date > end_date` or an unsupported band list → `400`. Malformed
JSON or an out-of-range `lat`/`lon` → `422` (Pydantic field validation).

### `POST /upload`
Multipart upload of a Sentinel-2 L2A GeoTIFF. **Validates only — does not
run SR.** Requires exactly the four supported bands (`B04`, `B03`, `B02`,
`B08`, order-independent — `frame.preprocessing.validate_bands` reorders),
a 128×128 pixel grid (the one proven patch size), and a valid CRS.

Form fields: `file` (the GeoTIFF), `input_scale` (`"raw_digital_number"` or
`"reflectance"`, forwarded unchanged to `frame.preprocessing.preprocess_rgbn`
at `/sr/run` time).

Response (200) includes `upload_id`, `band_names`, `width`/`height`,
`crs`, `resolution_m`. A non-GeoTIFF file, a missing/incompatible CRS, a
missing required band, or the wrong pixel grid → `400`.

### `POST /sr/run`
Runs the full pipeline against a previously-uploaded scene:

```
read GeoTIFF -> frame.preprocessing.preprocess_rgbn
             -> frame.uncertainty.run_stochastic_uncertainty
                (the TTA ensemble call IS the frozen SEN2SR inference step;
                its mean prediction is the SR output, its std is the
                uncertainty -- one model-touching call, not two)
             -> frame.geospatial.derive_output_metadata / write_geotiff
                (SR mean -> GeoTIFF, per-band + overall std -> a second GeoTIFF)
             -> frame.consistency.run_consistency_diagnostics
```

Request: `{"upload_id": "...", "seed": 42}` (`seed` optional, defaults to
`FRAME_API_UNCERTAINTY_SEED`). Unknown `upload_id` → `404`.

Response (200) — every field the phase spec requires is present:

```json
{
  "job_id": "…",
  "status": "completed",
  "upload_id": "…",
  "model_name": "SEN2SRLite/NonReference_RGBN_x4",
  "input_shape": [4, 128, 128],
  "output_shape": [4, 512, 512],
  "resolution": {
    "native_resolution_m": 10.0,
    "sr_resolution_m": 2.5,
    "scale_factor": 4,
    "description": "SR-derived product — 2.5 m pixel grid"
  },
  "bands": ["B04", "B03", "B02", "B08"],
  "crs": "EPSG:32630",
  "uncertainty": {
    "label": "relative model-stability uncertainty",
    "scalar_summary": 0.0123,
    "scalar_summary_definition": "…",
    "overall_distribution": {"mean": 0.01, "std": 0.004, "p95": 0.02, "...": "..."},
    "n": 6,
    "seed": 42,
    "transform_names": ["identity", "flip_h", "..."],
    "disclaimer": "This is a relative, architecture-conditioned model-stability proxy … NOT a calibrated probability of error … NOT the upstream LAM explainability tool …"
  },
  "self_consistency": {
    "downsample_rmse": 0.004,
    "ndvi_discrepancy_mean_abs": 0.01,
    "b08_b04_ratio_discrepancy_mean_abs": 0.02,
    "note": "Compares the SR output against its own LR input only -- not a ground-truth accuracy check."
  },
  "metadata": {
    "inference_seconds": 0.16,
    "device": "cuda",
    "output_geospatial": {"crs": "EPSG:32630", "transform": ["..."], "bounds": ["..."], "width": 512, "height": 512},
    "preprocessing_mask_coverage": 1.0
  },
  "scientific_caveats": ["…three fixed caveat strings, see Scientific terminology below…"],
  "artifacts": {"sr_geotiff": "/…/sr_mean.tif", "uncertainty_geotiff": "/…/uncertainty.tif"},
  "created_at": "2026-…Z"
}
```

Reproducibility fields recorded on every run: `model_name`, `seed`,
`transform_names` (the exact TTA ensemble members used), `device`,
`inference_seconds`, `created_at`. No silent randomness — the TTA seed is
always explicit (request-supplied or the configured default), never
implicit.

### `POST /analysis/ndvi`
Runs the Phase 6 NDVI comparison/uncertainty-weighted analysis against a
completed `/sr/run` job's saved tensors (no re-inference). Request:
`{"job_id": "..."}`. Unknown `job_id` → `404`.

In addition to the scalar summary (`comparison`, `uncertainty_weighted_summary`),
the response's `artifacts` field lists three single-band NDVI GeoTIFFs
written alongside the summary — `native_ndvi_geotiff` (10 m grid),
`sr_ndvi_geotiff` (2.5 m pixel grid), `ndvi_diff_geotiff` (`|SR NDVI
downsampled to the native grid − native NDVI|`, 10 m grid). These are
`frame.analysis`'s own already-computed arrays (`report.native_ndvi.ndvi`,
`report.sr_ndvi.ndvi`, `report.ndvi_comparison.absolute_difference_map`) —
no new NDVI computation was added; Phase 7 simply hadn't persisted them as
files yet. *(Phase 8 addition, for the frontend's NDVI view.)*

### `GET /analysis/download/{analysis_id}/native-ndvi` / `.../sr-ndvi` / `.../ndvi-diff`
Streams the three NDVI GeoTIFFs above as `image/tiff`. `404` if the
analysis id is unknown. *(Phase 8 addition.)*

### `GET /sr/result/{job_id}`
Re-fetches a completed job's result — identical schema to `/sr/run`'s
response. `404` if unknown.

### `GET /sr/download/{job_id}` / `GET /uncertainty/download/{job_id}`
Streams the job's SR mean GeoTIFF (4 bands, `B04`/`B03`/`B02`/`B08`, 2.5 m
grid) or its uncertainty GeoTIFF (5 bands: one per-band std +
`overall_std`) as `image/tiff`. `404` if unknown.

### `GET /analysis/{analysis_id}`
Re-fetches a completed NDVI analysis by its own id — identical schema to
`/analysis/ndvi`'s response. `404` if unknown.

## Error handling

Every 4xx/5xx body has the same shape: `{"error": "<code>", "code":
"<code>", "detail": "<human-readable message>"}`. `detail` never contains
a Python traceback — a genuine internal failure (a real bug, not bad
input) returns a generic `500` with no exception text, and the real
exception is logged server-side only. See `frame/api/errors.py` for the
full exception → status-code mapping (every `frame.*` package's own
input-validation error family, e.g. `UnsupportedBandsError`,
`MissingCRSError`, `ShapeMismatchError`, maps to `400` — a caller-input
problem, not a server bug).

## Scientific terminology (read before building a frontend against this)

These constraints are enforced structurally by
`frame/tests/test_api_terminology.py`, independent of the running app —
any endpoint response is guaranteed to honor them:

- The SR output is **never** called "native 2.5 m Sentinel-2" or a "true
  2.5 m image." Sentinel-2's finest native band resolution is 10 m; it has
  never observed the ground at 2.5 m. The SR output is a **learned
  statistical inference resampled onto a 2.5 m pixel grid** — call it
  "SR-derived product — 2.5 m pixel grid" (`resolution.description` on
  every SR result).
- The Phase 5 uncertainty is **relative model-stability uncertainty**
  (`uncertainty.label`) — a test-time-augmentation ensemble spread, not a
  calibrated probability of error and not a confidence interval. Its
  `disclaimer` field says this explicitly on every response.
- LAM (`sen2sr/xai/lam.py`, upstream explainability/sensitivity) is a
  **different tool answering a different question** (which input pixels
  influence the output, via gradients on blurred input copies) and is
  **not** exposed by any endpoint in this phase, and never described as
  "uncertainty" anywhere in this codebase.
- Self-consistency diagnostics compare the SR output against its own
  low-resolution input — **not** a ground-truth accuracy check.

## Prototype behavior (read this first)

- **Synchronous execution.** `/sr/run` blocks until the pipeline finishes
  (typically well under a second on GPU for the one proven 128×128 RGBN
  patch size — see `experiments/baseline/README.md`). There is no
  polling/job-queue endpoint because there is no async execution to poll.
- **In-memory job registry.** `frame.api.services.storage.JobStore` is a
  plain Python dict-backed registry, one instance per running process.
  Restarting the API process discards all upload/job/analysis records
  (the artifacts on disk under `FRAME_API_WORKSPACE_DIR` remain, but are
  no longer reachable by id). This is intentional — adding a database is
  out of scope for an SIH demo backend.
- **Artifact lifecycle.** `/upload` writes the raw file to
  `$FRAME_API_WORKSPACE_DIR/uploads/`. `/sr/run` writes, per job, under
  `$FRAME_API_WORKSPACE_DIR/jobs/<job_id>/`: `sr_mean.tif`,
  `uncertainty.tif`, and three `.pt` tensors (`input_tensor.pt`,
  `sr_mean_tensor.pt`, `sr_std_tensor.pt`) that `/analysis/ndvi` reads
  back — no re-inference for analysis. `/analysis/ndvi` in turn writes
  three more single-band GeoTIFFs into the same job directory, named with
  the analysis id (`ndvi_native_<analysis_id>.tif`,
  `ndvi_sr_<analysis_id>.tif`, `ndvi_diff_<analysis_id>.tif`). Nothing is
  auto-deleted; cleanup of old runs is a manual/future concern, not
  implemented here.
- **Model caching.** The compiled SEN2SRLite model is loaded at most once
  per device per process (`frame.api.services.model.get_model`, a
  module-level cache keyed by resolved device string) — not on every
  request.
- **Scope: 4-band RGBN only.** This phase only supports the
  `B04`/`B03`/`B02`/`B08` path, matching every prior phase. All 10
  Sentinel-2 bands are out of scope here.
- **How a frontend consumes this.** Typical flow: `POST /aoi/preview` to
  validate a request shape → user supplies a real Sentinel-2 L2A GeoTIFF →
  `POST /upload` → `POST /sr/run` with the returned `upload_id` →
  render/download via `GET /sr/download/{job_id}` and
  `GET /uncertainty/download/{job_id}` → optionally `POST /analysis/ndvi`
  for the NDVI comparison. `GET /sr/result/{job_id}` and
  `GET /analysis/{analysis_id}` let a page reload without re-running
  anything.

## Testing

- `frame/tests/test_api.py` — HTTP-layer tests via FastAPI's `TestClient`,
  with the model dependency (`routes.get_model_callable`) and job store
  (`routes.get_store`) overridden to a small deterministic fake and a
  fresh in-memory store respectively. No network, no real weights.
- `frame/tests/test_api_errors.py`, `test_api_storage.py`,
  `test_api_model_service.py`, `test_api_pipeline.py`,
  `test_api_terminology.py` — focused unit tests per module.
- `frame/tests/test_api_integration.py` (marked `integration`, via
  module-level `pytestmark = pytest.mark.integration`) — the real,
  network-free end-to-end run: real cached model, real Baseline 0 scene
  (`experiments/baseline/outputs/input_tensor.pt` + its documented
  geospatial metadata), through every endpoint. Skips itself (does not
  fail) if the weights cache is absent, rather than downloading.

`pyproject.toml` sets `addopts = "-m 'not integration'"`, so the
`integration` marker is **excluded by default** — a plain `pytest`
invocation never touches the real model. Passing `-m integration`
explicitly on the command line overrides that default and selects only
the integration test.

```bash
sen2sr_venv/bin/python -m pytest frame/tests/ -q                # normal suite -- integration test excluded
sen2sr_venv/bin/python -m pytest frame/tests/ -m integration -v # only the real integration test
```
