# FRAME — Technical Specification

**SIH Problem Statement 26142 — Uncertainty-aware, geospatially consistent Sentinel-2 super-resolution mapping platform**

Status: **All planned phases complete (Phase 9, final integration and hardening, closed out this document's implementation).** This document's body is preserved as the original design rationale — the `[FACT]`/`[PROPOSAL]`/`[OPEN QUESTION]` tags below reflect the state of understanding *at the time each section was written*, not necessarily the final implementation (each `experiments/*/README.md` and `frame/*/README.md` is the authoritative record of what was actually built and verified for its own phase). For the final, audited statement of what this project does and does not establish scientifically, see `docs/FINAL_SCIENTIFIC_AUDIT.md`.

| Phase | Scope | Status |
|---|---|---|
| 0 | Baseline — unmodified upstream inference path, reproducible | ✅ |
| 1 | Preprocessing layer (`frame.preprocessing`) | ✅ |
| 2 | Geospatial handling (`frame.geospatial`) | ✅ |
| 3 | Spectral/self-consistency checks (`frame.consistency`) | ✅ |
| 4 | External-reference validation (`frame.validation`, opensr-test) | ✅ |
| 5 | Model-stability uncertainty (`frame.uncertainty`) | ✅ |
| 6 | Downstream NDVI demonstration (`frame.analysis`) | ✅ |
| 7 | FastAPI backend (`frame/api/`) | ✅ |
| 8 | React frontend (`frontend/`) | ✅ |
| 9 | Final end-to-end integration, reproducibility, and repository-wide hardening audit | ✅ |

No further phase is planned.

## How to read this document

Every substantive claim is tagged so design intent is never confused with verified fact:

- **[FACT]** — verified by reading the code/config/output in this repository as of this writing. Falsifiable by re-reading the cited file.
- **[PROPOSAL]** — a design decision for future phases. Not implemented. Open to revision.
- **[OPEN QUESTION]** — a decision deliberately left unresolved because it needs more evidence (a real benchmark, a stakeholder choice, a compute budget) before committing.

No numerical performance target, benchmark score, or uncertainty-calibration number appears anywhere in this document. Where a metric or threshold will eventually be needed, that is stated explicitly as a gap to fill later with measured data, not a number invented now.

---

## 1. Project Definition

### 1.1 Problem we solve

Sentinel-2 L2A imagery is free, globally available, and revisits every 5 days, but its native spatial resolution (10 m for RGB+NIR, 20 m for red-edge/SWIR) is too coarse for many field-level analysis tasks (individual farm plots, urban infrastructure, small-scale change detection). Very-high-resolution commercial imagery (Planet, Maxar) solves the resolution problem but is expensive, licensed per-scene, and not freely reproducible for a hackathon-scale or public-sector deployment.

**[FACT]** The `sen2sr` package already exists to close part of this gap: it is a pretrained deep-learning super-resolution (SR) system that upsamples Sentinel-2 imagery to a 2.5 m pixel grid (4× for 10 m bands, 8× for 20 m bands via a two-stage cascade). It is a third-party research package (`pyproject.toml`: `name = "sen2sr"`, `version = "0.8.5"`, authors Cesar Aybar and Julio Contreras, ESAOpenSR project) — not something we wrote. This repository's `LICENSE` file (CC0 1.0 Universal, present since the initial commit) governs the repository as distributed here; see Section 22 and the Phase 9 license/attribution audit (`docs/FINAL_SCIENTIFIC_AUDIT.md`) for the correction of an earlier, incorrect MIT badge/claim.

What is missing, and what this project (FRAME) adds, is everything a real deployment needs *around* that model:
- A way to know **when to trust** the SR output and when not to (uncertainty).
- A way to **verify** that "sharper-looking" is not the same as "quantitatively more accurate" (validation).
- A way to keep the output **geospatially usable** (correct CRS, affine transform, exportable as GeoTIFF) instead of a bare tensor.
- A way to turn a super-resolved raster into something a non-ML-specialist stakeholder can act on (downstream analysis, a UI, explainability).

### 1.2 What the system does

FRAME is a pipeline and (eventually) an application that:
1. Ingests Sentinel-2 L2A imagery for a user-specified area of interest (AOI) and date, or a user-uploaded raster.
2. Preprocesses it (band selection, reflectance normalization, cloud/no-data handling, tiling for large AOIs).
3. Runs the existing, unmodified `sen2sr` SR models to produce a 2.5 m-pixel-grid raster.
4. Enforces and checks spectral/radiometric consistency between the SR output and the original observation.
5. Preserves and re-attaches full geospatial referencing (CRS, affine transform, bounds) so the output is a valid GeoTIFF, not just an image.
6. Attaches a **per-pixel uncertainty estimate** to the SR output (design in Section 13; not yet implemented).
7. Runs a scientifically honest **validation protocol** distinguishing what can and cannot be proven without native 2.5 m ground truth (Section 11).
8. Offers lightweight, honest **downstream analytical demonstrations** (agriculture, urban, disaster/change) built on top of the SR + uncertainty output (Section 15).
9. Exposes this through an API and a frontend so a non-technical user can run the whole flow (Sections 16–17; not yet built).

### 1.3 What the system does NOT claim

- It does **not** claim to recover the true, physically-measured reflectance of the ground at 2.5 m resolution. The model performs learned statistical inference conditioned on 10 m/20 m input; it invents plausible high-frequency spatial detail, it does not observe it.
- It does **not** claim the underlying SR neural network architecture, training, or the Fourier hard-constraint mechanism as our contribution — that is 100% upstream `sen2sr` work (Section 24 makes this explicit).
- It does **not** claim any accuracy number (PSNR/SSIM/etc.) until an actual benchmark has been run and reported (Sections 11–12). No such number exists yet in this repository.
- It does **not** claim the uncertainty estimate (once implemented) is a calibrated, physically rigorous confidence interval. Section 13 is explicit that any tractable uncertainty method compatible with a frozen pretrained model gives a *relative*, *architecture-conditioned* uncertainty signal (useful for triage: "look here first"), not an absolute error bound.
- It does **not** claim that a 2.5 m-pixel-grid output is the same thing as a native 2.5 m sensor observation (see 1.4).
- It does **not** propose replacing the SR backbone with a different model. `sen2sr`/SEN2SRLite is the fixed backbone for this project unless a demonstrated requirement is found (Section 8).

### 1.4 The 2.5 m output grid vs. a native 2.5 m observation — a distinction that must never be blurred

This is the single most important scientific-honesty boundary in the whole project, and it recurs in Sections 6, 11, 20, and 24.

- A **native 2.5 m observation** would mean a sensor actually measured reflected radiance at 2.5 m ground sampling distance (GSD) — e.g., a real VHR satellite or aerial platform. Sentinel-2 does not do this; its finest native band resolution is 10 m.
- The SR model output is a raster **resampled onto a 2.5 m pixel grid**, where each pixel's value is a *statistical inference* produced by a neural network trained to hallucinate plausible sub-10 m spatial structure consistent with the coarse input and with patterns the network learned from its (upstream, not our) training data. The Fourier hard constraint (Section 9) guarantees the *low-frequency* content of that inference matches the real 10 m observation, but the *high-frequency* detail is generated, not measured.
- Concretely: **pixel count increases 16× (4× per axis), information content does not increase 16×.** The true information ceiling is set by the original 10 m/20 m acquisition plus whatever prior the network learned; no post-hoc processing (ours or upstream) can exceed that ceiling.
- Every output artifact this system produces (raster metadata, UI labels, exported GeoTIFF tags) must communicate resolution as **"2.5 m pixel grid, Sentinel-2 SR-derived"**, never as "2.5 m Sentinel-2 imagery" or "2.5 m resolution" unqualified, to avoid implying a sensor capability that does not exist.

---

## 2. Existing Foundation

### 2.1 What SEN2SR already provides — [FACT], verified by reading `sen2sr/`

| Capability | File | Verified behavior |
|---|---|---|
| Non-reference 4× SR (RGBN only) | `sen2sr/nonreference.py` | Wraps an SR model + a hard-constraint module; forward pass = `clamp(sr_model(x), min=0)` then hard-constraint correction. |
| Reference 2× SR (20 m → 10 m, RSWIR bands) | `sen2sr/referencex2.py` | Resamples all 10 bands to a common 10 m grid, runs SR, reconstructs a 10-band stack from native 10 m + SR'd 20 m bands. |
| Full reference 4× cascade (10 m+20 m → 2.5 m, all 10 bands) | `sen2sr/referencex4.py` | Composes: 2× fusion (20 m→10 m) → 4× RGBN SR → 4× SWIR fusion SR → hard constraint → reassembles the full Sentinel-2 10-band stack in native band order (B2…B12). |
| Spectral/radiometric hard constraint | `sen2sr/models/tricks.py` (`FourierHardConstraint`, `HardConstraint`) | FFT-domain low-pass/high-pass recombination: low frequencies forced to match the bicubic-upsampled LR observation, high frequencies taken from the SR output. Four filter kernels available (ideal, Butterworth, Gaussian, sigmoid). |
| Large-image tiling | `sen2sr/utils.py` (`predict_large`, `define_iteration`, `fix_lastchunk`) | Fixed 128×128 patch iteration with configurable overlap (default 32 px) and edge-aware cropping/stitching to avoid seam artifacts. |
| Explainability (Local Attribution Map) | `sen2sr/xai/lam.py` | Blur-ramp gradient attribution: perturbs the input at increasing blur scales, backprops a windowed gradient-magnitude objective, and reports a KDE saliency map, a Gini-based "complexity" scalar, and a blur-robustness curve. **This is a sensitivity/explainability tool, not an uncertainty estimator** (see Section 14). |
| Pretrained model distribution | via `mlstac` (external dep) from Hugging Face (`tacofoundation/sen2sr`) | Model variants: `SEN2SRLite` (main, all bands), `SEN2SRLite/NonReference_RGBN_x4` (10 m RGBN only), `SEN2SRLite/Reference_RSWIR_x2` (20 m→10 m), and a flagship `LDSR-S2`/`SEN2SR` latent-diffusion model that requires `mamba-ssm` + CUDA > 12. |
| Data acquisition path used in all upstream examples | `cubo` (external dep, STAC-based Sentinel-2 L2A cube fetch) | Not part of `sen2sr` itself; used only in README examples and our Baseline 0 script. |

Package metadata **[FACT]** (`pyproject.toml`): `sen2sr` itself only *requires* `tqdm`, `numpy`, `einops` at the packaging level, but `sen2sr/__init__.py` hard-requires `torch` and `timm` to be importable (raises `ImportError` otherwise) — those are expected to be provided by the environment, not declared as poetry dependencies. Supported Python: `>=3.10,<4.0`.

**[FACT]** There is **no** geospatial handling anywhere in the `sen2sr` package: no `rasterio`, `affine`, `CRS`/`EPSG`, `GeoTIFF`, `pyproj`, `geopandas`, or `rioxarray` reference anywhere in `sen2sr/` (verified by grep across the package). The package operates purely on plain `torch.Tensor` objects shaped `(C, H, W)` / `(B, C, H, W)`. Any georeferencing that survives into the output today is whatever the *caller* (e.g. `cubo`, which returns a geospatially-aware `xarray.DataArray`) chooses to carry forward — and the current Baseline 0 script does not, because it calls `.to_numpy()` during preprocessing, which discards the DataArray's CRS/affine metadata. Re-attaching that metadata is entirely our responsibility (Section 10).

**[FACT]** There is **no** quantitative image-quality metric implemented anywhere in `sen2sr` (no PSNR/SSIM/RMSE/SAM/ERGAS). This is confirmed both by reading the package and by the Baseline 0 README's own note to this effect.

### 2.2 What Baseline 0 proves — [FACT]

Location: `experiments/baseline/` (currently untracked in git — not yet committed).

- `run_baseline.py` drives the upstream `SEN2SRLite/NonReference_RGBN_x4` model exactly the way the project's own README documents: `mlstac` downloads/loads the model, `cubo` fetches a Sentinel-2 L2A cube, the tensor is reflectance-scaled (`/10000`) and NaN/Inf-cleaned, and one forward pass is run.
- It imports **no internal `sen2sr` module**, subclasses nothing, and modifies no file under `sen2sr/` — it is a pure consumer of the published package (stated in `experiments/baseline/README.md` and verified by reading the script: only `import cubo`, `import mlstac`, `import torch`, `import numpy`, `import matplotlib`, no `import sen2sr`).
- It is deterministic by construction: a fixed AOI (lat 39.49152740347753, lon −0.4308725142800361 — the same point used in the upstream README examples), a single-day STAC query window (2023-01-15 to 2023-01-16, chosen because it is cloud-free at this AOI, per the Phase 0 investigation) instead of the upstream README's wide date-range scan, and a fixed time index (0). No random seeding is required because nothing in the path is stochastic.
- A completed run's actual recorded metadata (`experiments/baseline/metadata/run_metadata.json`) confirms:
  - Input shape `(4, 128, 128)` at 10 m → output shape `(4, 512, 512)` at 2.5 m, scale factor exactly 4.0.
  - Bands `[B04, B03, B02, B08]` (Red, Green, Blue, NIR).
  - Inference time 0.1553 s, peak GPU memory 100.5 MiB on an NVIDIA RTX 3050 Laptop GPU, PyTorch 2.14.0+cu130.
- Four PNG visualizations and two raw tensors (`input_tensor.pt`, `sr_tensor.pt`) are saved alongside the metadata.

**What Baseline 0 explicitly is not** (stated in its own README and preserved here as fact, not our added interpretation): it is **not** an accuracy benchmark. No metric is computed because there is no reference high-resolution image to compare against in this run. It proves the *pipeline* (weights → scene → preprocessing → forward pass → output shape) works reproducibly — nothing about quantitative correctness.

### 2.3 What we will reuse unchanged

- The entire `sen2sr` package as a black-box dependency: `nonreference.py`, `referencex2.py`, `referencex4.py`, `models/tricks.py` (both hard-constraint classes), `utils.py` (`predict_large` tiling), and `xai/lam.py`.
- The `mlstac` weight-loading mechanism and the Hugging Face model artifacts it points to.
- The reflectance normalization convention (`/10000`, NaN/Inf → 0) already used by the upstream README and Baseline 0 — this is Sentinel-2 L2A's standard scaling and is not something to redesign.
- The `predict_large` tiling strategy (128×128 patches, configurable overlap) as the default tiling approach for AOIs larger than one patch, rather than inventing a new tiler.
- The Baseline 0 experiment script and its fixed AOI/date as the canonical smoke-test fixture for regression-checking that future preprocessing/postprocessing changes don't break the base inference path.

### 2.4 Explicit separation: upstream functionality vs. our additions

| Layer | Owner | Status |
|---|---|---|
| SR neural network architectures (CNN/Swin/Mamba baselines under `sen2sr/models/opensr_baseline/`) | Upstream ESAOpenSR | Unmodified, reused as-is |
| Fourier hard constraint | Upstream | Unmodified, reused as-is |
| Tiling for large images | Upstream | Unmodified, reused as-is |
| LAM explainability | Upstream | Unmodified, reused as-is; re-labeled to users as "sensitivity," not "uncertainty" (Section 14) |
| Pretrained weights | Upstream (Hugging Face, `tacofoundation/sen2sr`) | Unmodified, reused as-is |
| Ingestion/AOI selection UX | **FRAME** | New |
| Preprocessing (cloud/shadow masking, tiling orchestration, metadata capture) | **FRAME** | New, wraps upstream utilities |
| Geospatial re-attachment (CRS/affine/bounds → GeoTIFF) | **FRAME** | New — upstream provides none of this |
| Quantitative spectral consistency checks (beyond the existing hard constraint) | **FRAME** | New |
| Uncertainty estimation | **FRAME** | New — upstream provides none |
| Validation/benchmarking harness and metrics | **FRAME** | New — upstream provides none |
| Downstream analytical demos | **FRAME** | New |
| API + frontend | **FRAME** | New, not yet built |

No file under `sen2sr/` will be edited during this project. Where upstream behavior needs to be extended (e.g., attaching CRS metadata to a `predict_large` output), the extension will live in FRAME's own modules that call `sen2sr` as a library, never by patching `sen2sr` in place.

---

## 3. Final System Architecture

**[PROPOSAL]** — target architecture; none of the boxes below except "Existing SR Core" exist yet.

```
                                    ┌──────────────────────────────┐
                                    │           FRONTEND            │
                                    │  AOI select · preview · run   │
                                    │  compare · uncertainty view   │
                                    │  metrics · analysis · export  │
                                    └───────────────┬────────────────┘
                                                     │ HTTPS / JSON + file download
                                    ┌───────────────▼────────────────┐
                                    │             API LAYER          │
                                    │   FastAPI (Section 17)         │
                                    │   request validation, job      │
                                    │   orchestration, file serving  │
                                    └───────────────┬────────────────┘
                                                     │
   ┌─────────────────────────────────────────────────────────────────────────────────┐
   │                              FRAME PIPELINE (Python)                             │
   │                                                                                   │
   │  ┌────────────┐    ┌────────────────┐    ┌───────────────────────────────────┐  │
   │  │ INGESTION  │───▶│  PREPROCESSING │───▶│         SR CORE (EXISTING,        │  │
   │  │ STAC/cubo  │    │ band select    │    │         UNMODIFIED sen2sr)        │  │
   │  │ or user    │    │ normalize      │    │  nonreference / referencex2 /     │  │
   │  │ GeoTIFF    │    │ cloud/shadow   │    │  referencex4, hard constraint,    │  │
   │  │ upload     │    │ mask, align,   │    │  predict_large tiling             │  │
   │  │            │    │ tile           │    │                                    │  │
   │  └────────────┘    └────────────────┘    └────────────────┬──────────────────┘  │
   │                                                             │                    │
   │        ┌────────────────────────────────────────────────────┴───────┐           │
   │        ▼                                                             ▼           │
   │  ┌────────────────────┐                                  ┌───────────────────┐  │
   │  │ SPECTRAL CONSISTENCY│                                  │ GEOSPATIAL HANDLING│  │
   │  │ existing Fourier    │                                  │ CRS / affine /     │  │
   │  │ constraint (reused) │                                  │ bounds re-attach,  │  │
   │  │ + new quantitative  │                                  │ GeoTIFF export     │  │
   │  │ checks (Section 9)  │                                  │ (Section 10)       │  │
   │  └──────────┬──────────┘                                  └─────────┬──────────┘  │
   │             │                                                        │             │
   │             └───────────────────────┬────────────────────────────────┘             │
   │                                      ▼                                              │
   │                          ┌──────────────────────┐                                   │
   │                          │ UNCERTAINTY ESTIMATION│                                   │
   │                          │ (Section 13, one       │                                   │
   │                          │ candidate method,      │                                   │
   │                          │ not yet implemented)    │                                   │
   │                          └──────────┬─────────────┘                                   │
   │                                     ▼                                                 │
   │                          ┌──────────────────────┐                                     │
   │                          │      VALIDATION       │                                     │
   │                          │ synthetic-degradation  │                                     │
   │                          │ + external-reference    │                                     │
   │                          │ + deployment checks     │                                     │
   │                          │ (Section 11)             │                                     │
   │                          └──────────┬─────────────┘                                     │
   │                                     ▼                                                    │
   │                          ┌──────────────────────┐                                        │
   │                          │ DOWNSTREAM APPLICATIONS│                                       │
   │                          │ agriculture / urban /   │                                      │
   │                          │ disaster (Section 15)    │                                     │
   │                          └──────────────────────┘                                         │
   └───────────────────────────────────────────────────────────────────────────────────┘
```

Component responsibilities, one line each:

- **Ingestion** — resolve an AOI+date into Sentinel-2 L2A band arrays (via STAC), or accept a user-supplied georeferenced raster.
- **Preprocessing** — turn raw bands into a model-ready tensor while preserving everything needed to reconstruct geospatial context afterward.
- **SR core** — the unmodified upstream `sen2sr` inference path.
- **Spectral consistency** — reuse the existing hard constraint (it already runs inside the SR core's wrapper classes) and add independent, quantitative post-hoc checks.
- **Geospatial handling** — the part upstream does not provide at all; re-attaches CRS/affine/bounds and writes valid GeoTIFFs.
- **Uncertainty** — a new module producing a per-pixel confidence signal alongside the SR raster.
- **Validation** — an offline/CI-adjacent benchmarking harness, not part of the live request path.
- **Applications** — thin analytical layers consuming the SR + uncertainty output.
- **API** — orchestrates the above and exposes them over HTTP.
- **Frontend** — the human-facing workflow (Section 16).

---

## 4. Data Flow

**[PROPOSAL]** — end-to-end trace for one user-triggered SR request.

```
[1] INPUT
    AOI (lat/lon + edge size, or bbox) + date, OR user-uploaded GeoTIFF
        │
        ▼
[2] PREPROCESSING
    - fetch bands via STAC (cubo) or read the uploaded raster
    - validate CRS, resolution, band set (Section 5)
    - reflectance-scale (/10000), clip/clean no-data & NaN/Inf
    - cloud/shadow mask from L2A SCL band (or user-provided mask)
    - tile into 128×128 patches if AOI exceeds native patch size
    - RECORD: source CRS, affine transform, bounds, acquisition timestamp,
      band order, nodata mask — this metadata must survive to step [7]
        │
        ▼
[3] SUPER-RESOLUTION (unmodified sen2sr)
    - run the selected variant (Section 8) per patch (or predict_large for
      multi-patch AOIs)
    - hard constraint applied internally by the model wrapper (existing)
        │
        ▼
[4] POSTPROCESSING
    - re-stitch tiles (existing predict_large logic)
    - re-project pixel grid metadata: new affine = old affine scaled by 1/4
      (or 1/8 for the two-stage cascade), same CRS, bounds unchanged
    - re-apply/propagate the no-data mask at the new resolution
        │
        ▼
[5] SPECTRAL CONSISTENCY CHECKS (new, Section 9)
    - quantitative checks that the hard-constrained output's low-frequency
      content matches the LR observation within an acceptable band
        │
        ▼
[6] UNCERTAINTY ESTIMATION (new, Section 13)
    - produce a per-pixel (or per-tile) confidence/uncertainty raster,
      co-registered with the SR output
        │
        ▼
[7] VALIDATION (offline/asynchronous, not on the live request path — Section 11)
    - synthetic-degradation and/or external-reference benchmarks run
      separately against held-out data, not against the live user request
        │
        ▼
[8] ANALYSIS (optional, per Section 15)
    - lightweight downstream computation (NDVI, edge/segmentation-based
      built-up proxy, before/after differencing) on the SR output
        │
        ▼
[9] EXPORT
    - GeoTIFF (SR bands), GeoTIFF or PNG (uncertainty), PNG (visual preview),
      JSON (run metadata + any computed metrics)
```

The critical invariant carried through steps 2→9 is **georeferencing continuity**: CRS and affine transform are captured once at ingestion and mathematically propagated (never re-derived or guessed) through every resolution change.

---

## 5. Input Contract

**[PROPOSAL]** — what the FRAME backend will accept. Nothing here is implemented; this defines the target contract.

- **Sentinel-2 bands.**
  - Minimum: the four 10 m bands `B04, B03, B02, B08` (Red, Green, Blue, NIR) — required for the `NonReference_RGBN_x4` path used by Baseline 0.
  - Optional/extended: the full 10-band L2A set `B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12` (10 m + 20 m) — required for the full reference cascade (`referencex4.py`).
  - Band order must match what each upstream wrapper expects internally (see Section 2.1 table); FRAME's ingestion layer is responsible for reordering, not the user.
- **Raster format.** Sentinel-2 L2A source via STAC (ingested through `cubo`, as in Baseline 0), or a user-uploaded GeoTIFF/COG containing the required bands as separate bands or files. Non-georeferenced rasters (plain PNG/JPEG) are out of scope for the input contract — the platform's entire value proposition depends on geospatial validity, so an ungeoreferenced input cannot be honestly processed end-to-end.
- **Expected spatial resolution.** 10 m per pixel for the RGBN path; 10 m and 20 m per pixel (co-registered, same AOI) for the full cascade. Inputs at other native resolutions are rejected, not silently resampled to fit — resampling before SR would conflate two different degradations and invalidate any later benchmark.
- **Reflectance representation.** Integer digital numbers scaled by 10,000 (Sentinel-2 L2A's standard convention, `/10000` → float32 reflectance in ~[0, 1]), matching upstream's own preprocessing. Inputs already provided as float reflectance must be explicitly flagged as such at ingestion (no silent double-scaling).
- **CRS requirements.** Input must carry a valid, readable CRS (any EPSG code `cubo`/STAC or `rasterio` can resolve). No CRS ⇒ hard rejection at ingestion, with a clear error — this is a hard boundary because the entire geospatial-consistency guarantee (Section 10) depends on a known source CRS.
- **Metadata requirements.** Acquisition date/time, band list with wavelength/resolution association, source CRS, affine transform (or equivalent bounds+resolution), and — where available — the L2A Scene Classification (SCL) band or an equivalent cloud/shadow mask.
- **No-data / cloud handling.** Pixels marked no-data, or classified as cloud/cloud-shadow/snow in SCL (when available), are masked before SR inference, not passed through as if they were valid reflectance. The corresponding output pixels are flagged in the output's no-data/uncertainty layer rather than silently super-resolved as if trustworthy (see Section 7).

**[OPEN QUESTION]** Whether to support Sentinel-2 L1C (top-of-atmosphere) input in addition to L2A. Upstream examples and Baseline 0 use L2A exclusively; L1C would need its own normalization convention and is out of scope unless a concrete requirement emerges.

---

## 6. Output Contract

**[PROPOSAL]**

- **SR raster.** Multi-band GeoTIFF, float32 reflectance (or scaled integer, decision open — see below), same band semantics as the input (RGBN or full 10-band), values clamped ≥ 0 by the existing upstream wrapper.
- **Resolution.** 2.5 m pixel grid for the RGBN 4× path; 2.5 m pixel grid for the full cascade (20 m 2× to 10 m, then 4× to 2.5 m). Always labeled per Section 1.4's distinction — "2.5 m pixel grid, SR-derived," never bare "2.5 m."
- **Bands.** Either 4 bands (RGBN path) or 10 bands in native Sentinel-2 order (full cascade path), matching whichever upstream variant was used, tagged with band names in file metadata.
- **CRS.** Identical EPSG code as the input — SR changes pixel size, not projection.
- **Affine transform.** Derived deterministically from the input's affine transform: pixel width/height divided by the scale factor (4 or 8), origin (top-left corner coordinate) unchanged, rotation/shear terms unchanged.
- **Bounds.** Identical geographic bounding box as the input AOI (SR increases pixel density inside the same footprint, it does not extend the footprint).
- **GeoTIFF requirements.** Valid CRS tag, valid affine transform (`GeoTransform`), correct band count/order/naming, explicit no-data value, and enough tags (model name/version, source acquisition date, FRAME pipeline version) to make a downloaded file self-describing without the original request context.
- **Uncertainty output.** A co-registered raster (same CRS, affine transform, bounds, and pixel grid as the SR output) with one band per uncertainty channel (e.g., one scalar confidence band, or one per output band depending on the method chosen in Section 13), exported as a separate GeoTIFF so it can be loaded and overlaid in any GIS tool.
- **Visualization output.** RGB-stretched PNG previews of both the SR result and the uncertainty layer (e.g., as a heatmap), for use in the frontend and in reports — visual-only, not authoritative data.

**[OPEN QUESTION]** float32 vs. scaled-uint16 for the exported SR GeoTIFF. Float32 is simpler and lossless relative to the model's own output; scaled-uint16 (matching the `/10000` L2A convention) halves file size and keeps parity with how Sentinel-2 data is normally distributed. Decide once real export file sizes are measured (Section 19, geospatial-export experiment).

---

## 7. Preprocessing Design

**[PROPOSAL]** — builds on, but extends, what Baseline 0 does today.

- **Band selection.** Determined by which SR variant is requested (RGBN-only vs. full 10-band); FRAME's ingestion layer selects and reorders bands to match the exact order each upstream wrapper expects (documented per-wrapper in Section 2.1), rather than requiring the caller to know upstream's internal ordering.
- **Normalization.** Reuse the existing `/10000` reflectance scaling and NaN/Inf → 0 cleanup exactly as Baseline 0 does it — this is not something to redesign, it is the convention the pretrained weights were trained against.
- **Alignment.** For the full 10-band cascade, 10 m and 20 m bands must be co-registered to the same AOI/grid before being handed to `referencex4.srmodel` (which itself performs the 20 m→10 m resample as its first internal step) — FRAME's job is only to guarantee both band groups come from the *same* acquisition/AOI, not to reimplement resampling upstream already does internally.
- **Cloud/shadow masking.** Derive a mask from the Sentinel-2 L2A SCL band (cloud, cloud shadow, cirrus, snow classes) when available; when the user supplies a raster without SCL, accept an optional user-provided mask, or fall back to flagging the entire tile as "mask unavailable" rather than fabricating a mask. Masked pixels are excluded from SR confidence claims and flagged through to the uncertainty/no-data output layer — they are not silently super-resolved as if clean.
- **Tiling.** Reuse `sen2sr.utils.predict_large` (128×128 patches, configurable overlap) as the default; FRAME's preprocessing layer is responsible only for building the correctly-shaped tensor `predict_large` expects and for carrying tile-to-geolocation bookkeeping so each output tile can be placed back at the correct real-world coordinates.
- **Invalid pixels.** No-data and masked-cloud pixels are tracked as a boolean/categorical mask parallel to the reflectance tensor throughout preprocessing, SR, and postprocessing — never silently zero-filled and forgotten, since a zero-filled pixel is indistinguishable from a real dark-reflectance pixel unless the mask travels with it.
- **Metadata preservation.** Every value needed to reconstruct geospatial and provenance context (source CRS, affine transform, bounds, acquisition timestamp, band list, cloud-mask coverage %, SR variant used) is captured once at ingestion into a single run-metadata record (following the same spirit as Baseline 0's `run_metadata.json`, extended with the geospatial fields upstream currently drops).

---

## 8. Super-Resolution Design

**[FACT + DECISION]**

We use **`SEN2SRLite/NonReference_RGBN_x4`** as the baseline SR variant, exactly as validated in Baseline 0 (Section 2.2), for these reasons:

1. **Already proven working end-to-end in this repository.** Baseline 0 is a real, reproducible run against this exact variant — it is the only SR path in this project with actual measured runtime/memory numbers behind it.
2. **Lowest dependency and compute footprint.** It needs only the 4 native 10 m bands (no 20 m fusion stage), and — per the upstream README — does not require `mamba-ssm` or CUDA > 12, unlike the flagship `LDSR-S2`/`SEN2SR` latent-diffusion model. This matters for a hackathon-timeline deployment on modest/shared GPU hardware (Baseline 0 ran comfortably on a laptop RTX 3050, 100.5 MiB peak memory).
3. **Sufficient for the core demo scope.** RGBN 4× (10 m → 2.5 m) already covers the bands most downstream visual/analytical demonstrations need (true color + NIR for vegetation indices).
4. **Extensible without switching backbones.** The full 10-band cascade (`referencex4.py`) is available in the *same* upstream package if a later phase needs SWIR/red-edge bands — this is a configuration choice within `sen2sr`, not a new model integration.

**We do not propose replacing this with a different SR architecture.** No requirement observed so far (accuracy shortfall, missing band support, licensing issue) justifies introducing a second model family. If validation (Section 11) later reveals the RGBN-only variant is inadequate for a specific downstream task, the first escalation path is the *already-available* full 10-band cascade within `sen2sr`, not a new external model.

**[OPEN QUESTION]** Whether Phase 2+ needs the full `referencex4` cascade (for SWIR-band applications, e.g. burn-scar/disaster mapping) — deferred until a downstream-application experiment (Section 19) demonstrates a concrete need.

---

## 9. Spectral Consistency

### 9.1 Existing Fourier hard constraint — [FACT]

Implemented in `sen2sr/models/tricks.py` as `FourierHardConstraint` (parametric filter shape) and `HardConstraint` (precomputed mask, optional band subsetting). Mechanism, precisely:

1. The LR input is bicubically upsampled (with antialiasing) to the SR output's pixel size.
2. Both the upsampled LR and the raw SR output are transformed to the frequency domain (`fft2`, shifted to center the zero frequency).
3. A low-pass mask (radius = `min(H,W)//scale_factor`, shape selectable: ideal / Butterworth / Gaussian / sigmoid) selects the **low-frequency component from the LR observation** and the **high-frequency component from the SR output**.
4. The two are recombined and inverse-transformed back to the spatial domain.

Effect: the SR output's coarse-scale radiometry is mathematically locked to what the sensor actually measured; only fine spatial detail is allowed to come from the network's inference. This is already wired into every upstream `srmodel` wrapper (`nonreference.py`, `referencex2.py`, `referencex4.py`) and runs on every inference — it is not optional and not something we need to re-implement.

This constraint is **structural/architectural**, not diagnostic: it changes what the model outputs, but by itself it produces no number we can report, log, or threshold against. That is the gap Section 9.2 addresses.

### 9.2 New quantitative checks — [PROPOSAL]

We will add independent, *post-hoc* measurements that verify the hard constraint is behaving as intended on a given real output, without assuming it always will (a runtime check, not a re-implementation of the constraint itself):

- **Downsample-consistency check.** Downsample the SR output back to the input's native resolution (matching the model's own scale factor) and compare it against the original LR input band-by-band. Report a per-band and per-tile discrepancy statistic (not yet a named threshold — see Section 12 for candidate metrics). Large discrepancy is a signal something in the pipeline (not necessarily the model) went wrong — e.g., a band-order mismatch, a masking bug, or a numerically degenerate tile.
- **Per-band spectral shape check.** Compare inter-band ratios (e.g., NDVI-like ratios, or simple B08/B04) between the LR input and the downsampled SR output, to catch cases where the low-frequency match is good in aggregate but a specific band has drifted (the hard constraint operates per-band already, but a checksum independent of the model's own math is valuable).
- **Cross-tile consistency check.** For AOIs requiring `predict_large` tiling, measure discrepancy in the overlap regions between adjacent tiles' outputs before blending, to catch seam artifacts.
- **Flagging, not gatekeeping (initially).** These checks produce a report attached to the run metadata and (eventually) surfaced in the UI; they do not silently discard or "fix" an output. Turning a check into an automatic rejection rule is a later decision once real discrepancy distributions have been observed on real data.

We deliberately do **not** invent numeric thresholds ("discrepancy must be below X") in this document — a threshold requires an empirical distribution from real runs (Section 11) to be meaningful rather than arbitrary.

---

## 10. Geospatial Consistency

**[PROPOSAL]** — this entire section is new work; Section 2.1 already established upstream provides none of it.

Goal: every pixel in the SR output must remain traceable to a real-world coordinate with the same reliability as the input, through every stage of the pipeline.

- **CRS.** Captured once at ingestion (from the STAC item or the uploaded raster's own CRS tag) and carried as an explicit value in the run's metadata object — never re-derived, guessed, or defaulted. SR changes pixel size, not projection, so the output CRS is always identical to the input CRS.
- **Affine transform.** The output transform is derived arithmetically from the input transform: `transform_out = transform_in * Affine.scale(1/scale_factor)` conceptually — pixel width and height divide by the scale factor (4, or the compounded value for a multi-stage cascade), rotation/shear terms are preserved unchanged, and the origin (top-left world coordinate) is preserved unchanged. This is a deterministic recomputation, not something inferred from the output image content.
- **Bounds.** The geographic bounding box of the output equals the geographic bounding box of the input; only pixel density inside that box changes. This is asserted as an invariant and checked, not assumed silently.
- **Pixel size.** Explicitly recomputed and stored (e.g., 2.5 m) rather than left implicit; this value is what drives the "2.5 m pixel grid" labeling from Section 1.4.
- **Tiling correctness.** Because `predict_large` tiles in *pixel* space, each tile's output placement must be mapped back to *world* coordinates using the same arithmetic as above, applied per-tile-offset, so that tile boundaries line up exactly with no gaps or overlaps in the final georeferenced mosaic — this bookkeeping is entirely FRAME's responsibility since `predict_large` itself only returns a pixel-space tensor.
- **Where this lives in code (proposed, not built).** A thin `geospatial` module that wraps a raster-plus-metadata object (candidate library: `rasterio` for CRS/affine/GeoTIFF I/O) around the plain tensors that flow through the `sen2sr` calls — the tensors handed to `sen2sr` itself remain plain, unmodified `torch.Tensor` objects exactly as upstream expects; the geospatial wrapper lives entirely in FRAME's own code, before and after the `sen2sr` call.
- **Validation of this design itself.** A dedicated experiment (Section 19, "Baseline + geospatial export") re-runs Baseline 0's exact AOI/scene, but wraps it with geospatial metadata capture and writes a real GeoTIFF, then verifies in a GIS tool (or `rasterio`/`gdalinfo`) that the exported file's CRS/bounds/pixel size are correct against the known AOI — this is a concrete, checkable acceptance test, not an aspiration.

---

## 11. Validation Strategy

This is the most important section, and the one most vulnerable to overclaiming. The structuring principle: **be explicit about what ground truth is (and is not) available for each benchmark type**, because the central scientific problem in SR validation is exactly this — see 11.4.

### 11.A Synthetic / degradation benchmark

- **Input.** A real high(er)-resolution image (e.g., a 10 m Sentinel-2 band, or an aerial/VHR image if available) that is *deliberately degraded* (downsampled, blurred, resampled) to simulate what the SR model's expected input looks like.
- **Target/reference.** The original, undegraded image — used as ground truth because *by construction* we know exactly what the true high-resolution content was, since we created the degradation ourselves.
- **Procedure.** Take image `Y` at resolution `R`. Degrade it (downsample by the model's scale factor, e.g. 4×) to get a synthetic LR input `X_synth` at resolution `4R`. Run `X_synth` through the SR model to get `Ŷ`. Compare `Ŷ` against the *original* `Y` (which is real, measured data, not model output).
- **Metrics.** PSNR, SSIM, RMSE, SAM, ERGAS (Section 12) computed between `Ŷ` and `Y`, since a genuine, real-pixel reference exists here.
- **What conclusion can legitimately be drawn.** This measures how well the model reconstructs a *known* degradation of *real* imagery — a meaningful, standard SR-evaluation protocol. **It does not** measure how well the model performs on the *actual* degradation gap between real Sentinel-2 10 m data and real 2.5 m ground truth, because the synthetic degradation (however carefully chosen) is a model of the true sensor/optics/atmosphere gap, not the gap itself. Results from this benchmark characterize the model's *reconstruction competence under a controlled, known blur/downsample operator* — a necessary but not sufficient validation.

### 11.B External high-resolution reference benchmark (if practical)

- **Input.** A real Sentinel-2 L2A scene (10 m/20 m, unmodified, real sensor data — no synthetic degradation).
- **Target/reference.** An independently-sourced, genuinely higher-resolution image of the *same location and a close-by date* — e.g., freely available aerial orthophotos, an open VHR dataset, or (if licensing and budget permit) a commercial VHR tasking. This is the only benchmark type in this document that could, in principle, validate the model against a real 2.5 m-class observation.
- **Procedure.** Co-register the reference image to the Sentinel-2 scene's CRS/grid; run the real Sentinel-2 input through the SR model; compare the SR output against the resampled/aligned reference at the reference's native resolution (or the SR output's resolution, whichever alignment introduces less additional resampling error).
- **Metrics.** Same image-quality metrics as 11.A, but interpretation must account for: (a) acquisition date mismatch (ground cover can change between the two acquisitions), (b) atmospheric/illumination differences between sensors, (c) co-registration error, which is itself a source of apparent "error" unrelated to the SR model.
- **What conclusion can legitimately be drawn.** This is the *closest* this project can get to validating against a real 2.5 m-class ground truth, but every number from it carries the caveats above baked in — it validates "the model's output resembles a real higher-resolution image of the same place," with acknowledged confounds, not "the model recovers the true reflectance."
- **Practicality caveat.** Whether this benchmark is achievable within project time/budget depends on finding a suitable, appropriately-licensed reference dataset for at least one AOI — this is explicitly marked **[OPEN QUESTION]**, to be resolved by a scoped data-sourcing task before committing engineering effort to the co-registration pipeline it requires.

### 11.C Real Sentinel-2 deployment evaluation (no exact HR ground truth)

- **Input.** Any real Sentinel-2 L2A scene a user actually submits in deployment — no reference image of any kind exists for these.
- **Target/reference.** **None exists.** This is the central, unavoidable limitation of Sentinel-2 SR in general, stated plainly rather than worked around: *there is no native 2.5 m Sentinel-2 ground truth, anywhere, because Sentinel-2 has never observed at 2.5 m.* Any claim of "accuracy" for a live deployment request is therefore, by construction, unverifiable against direct pixel-level ground truth.
- **Procedure.** Since no reference-based metric is possible, this evaluation relies entirely on **no-reference signals**: the spectral-consistency checks from Section 9.2 (does the output remain faithful to the real LR observation it was derived from?), the uncertainty output from Section 13 (does the model itself flag low-confidence regions?), and no-reference image-quality metrics (Section 12) that assess internal statistical plausibility without needing a second image.
- **What conclusion can legitimately be drawn.** Only relative, self-consistency conclusions: "this output is internally consistent with its input," "this region is flagged as low-confidence," "this output resembles the *statistical distribution* of natural imagery the no-reference metric was calibrated on." **None of this proves geometric/radiometric accuracy at 2.5 m** — it can only proves the absence of certain failure modes, never the presence of correctness. This distinction must be stated to every user of the deployed system, not just documented internally (see Section 16, UX).

### 11.D The absence of native 2.5 m Sentinel-2 ground truth — explicit treatment

This is restated deliberately because it is the fact that makes 11.C unavoidable: there is no dataset, public or private, of true 2.5 m-GSD Sentinel-2 sensor measurements, because the Sentinel-2 instrument's finest band is 10 m. Every "high-resolution reference" available to any Sentinel-2 SR project (this one included) is either (a) a synthetic degradation of some other real image (11.A) or (b) a different sensor entirely, with its own date, geometry, and radiometric characteristics (11.B). This is not a gap specific to FRAME or to `sen2sr` — it is structural to the Sentinel-2 SR problem, and the validation strategy above is designed around acknowledging it rather than concealing it behind a single confident-sounding accuracy number.

---

## 12. Metrics

For each metric: what it measures, how it will be used here, and its specific limitation for this project.

- **PSNR (Peak Signal-to-Noise Ratio).** Pixel-wise fidelity in dB, sensitive mainly to large-magnitude errors. Usable in 11.A/11.B where a real reference exists. **Limitation:** correlates poorly with perceptual/structural quality, and is meaningless without a reference (unusable for 11.C).
- **SSIM (Structural Similarity Index).** Captures structural/luminance/contrast similarity, closer to perceptual quality than PSNR. Usable in 11.A/11.B. **Limitation:** still a reference-based metric (unusable for 11.C); tuned for natural photographic imagery, not validated for multispectral reflectance statistics.
- **RMSE (Root Mean Squared Error).** Simple, interpretable-in-reflectance-units error magnitude. Usable in 11.A/11.B. **Limitation:** same reference-dependency as PSNR; sensitive to outliers (e.g., cloud-edge artifacts) unless masked pixels are excluded first.
- **SAM (Spectral Angle Mapper).** Measures spectral *shape* similarity between two pixel spectra independent of overall brightness — directly relevant to multispectral SR since it isolates whether inter-band relationships (e.g., vegetation "shape") are preserved. Usable in 11.A/11.B, only where ≥2 bands are compared. **Limitation:** insensitive to absolute magnitude errors (a uniformly-scaled spectrum passes SAM even if absolute reflectance is wrong) — must always be reported alongside a magnitude-sensitive metric, never alone.
- **ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).** A pansharpening/fusion-quality metric that aggregates per-band RMSE, normalized and scaled by the resolution ratio — standard in the remote-sensing SR/fusion literature specifically because it accounts for the resolution change, unlike PSNR/SSIM which were designed for same-resolution image comparison. Usable in 11.A/11.B. **Limitation:** like the others, requires a real reference.
- **No-reference metric(s), for 11.C.** Candidates: no-reference perceptual quality estimators (e.g., BRISQUE-family or learned no-reference IQA models), plus the spectral-consistency checks defined in Section 9.2 (downsample-consistency discrepancy) as a domain-appropriate no-reference signal specific to SR (rather than a generic photographic-quality score). **Limitation, stated plainly:** a no-reference score reflects *statistical plausibility relative to a training distribution* (for a learned no-reference IQA model) or *self-consistency with the LR input* (for the downsample-consistency check) — neither is evidence of ground-truth accuracy, and both must be labeled as such wherever shown to a user.

**No single metric, and no combination of metrics in this list, constitutes proof of scientific validity on its own.** Reference-based metrics (PSNR/SSIM/RMSE/SAM/ERGAS) are only as trustworthy as the reference they're computed against (Section 11.A/11.B caveats apply in full); no-reference metrics never had a ground-truth reference to be trustworthy against in the first place. The validation report for this project (once produced) must present metrics per benchmark type (11.A/11.B/11.C) side by side with their respective caveats, never as a single unqualified score.

---

## 13. Uncertainty Design

Comparing candidate approaches against the constraint that the SR backbone is a **frozen, pretrained** model we do not intend to retrain (Section 2.3/2.4 — reuse unchanged).

| Approach | Compatible with frozen pretrained `sen2sr`? | Implementation difficulty | Compute cost | Expected usefulness | Limitations |
|---|---|---|---|---|---|
| **MC Dropout** | Only if the pretrained architecture contains dropout layers active-able at inference time — **[OPEN QUESTION]**, not yet verified against the actual `sen2sr/models/opensr_baseline/` architectures (CNN/Swin/Mamba). If absent, this approach is not applicable without retraining, which is out of scope. | Low, *if* applicable (toggle dropout to train-mode at inference, run N forward passes, take variance) | Moderate — N× inference cost (N typically 10–30) per request | Per-pixel variance map correlates with regions where the network itself is "unsure" under its own stochastic regularization — a well-established, if imperfect, uncertainty proxy in the SR literature | Only as good as the dropout the architecture happens to have (not designed as an uncertainty mechanism); variance can be poorly calibrated; no guarantee dropout materially affects output for a model trained deterministically |
| **Deep ensemble** | Yes, always compatible — does not require modifying `sen2sr` at all, only running the *same* (or several different) pretrained variant(s)/checkpoints in parallel | Low conceptually, moderate operationally (need multiple distinct trained checkpoints, which may not all be available for every variant) | High — full N× inference cost for N ensemble members, N× GPU memory if run concurrently | Generally the most reliable, best-studied uncertainty-estimation approach; disagreement between ensemble members is a strong empirical uncertainty signal | Requires multiple genuinely different trained models — if only one checkpoint per variant exists (needs verifying against what `mlstac`/Hugging Face actually hosts), this collapses to zero diversity and produces a degenerate (near-zero) uncertainty signal |
| **Stochastic inference (input perturbation / test-time augmentation ensembling)** | Yes, fully compatible with a frozen model — run the same model multiple times on slightly perturbed/augmented versions of the same input (small crops, flips, rotations, minor noise) and measure output variance | Low — no model internals touched at all | Moderate — N× inference cost, same order as MC dropout | A safe, architecture-agnostic uncertainty proxy; also naturally exposes the model's sensitivity to input framing, which overlaps usefully with the LAM robustness signal already computed upstream | Measures sensitivity to *input perturbation*, which is a different (and weaker) notion of uncertainty than "how likely is this pixel to be wrong" — useful as a triage signal, not a calibrated error bound |
| **Learned uncertainty head** | No — requires adding and training a new output head on top of (or fine-tuning) the pretrained network, which contradicts the "reuse unchanged" principle for this phase | High — needs a training pipeline, a loss function (e.g., heteroscedastic NLL), and labeled/paired data to train against, none of which currently exist in this repository | Training cost is substantial; inference cost afterward is cheap (single extra forward pass) | Potentially the most accurate and best-calibrated *if* trained well, since it can learn where the frozen backbone systematically struggles | Requires the training data problem, and the ground-truth problem (Section 11.D), to be solved *first* — i.e., it depends on outputs from other unresolved parts of this same spec; also risks becoming an entire second ML project unless deliberately scoped down |

### Recommendation

**[PROPOSAL, not implemented]** — recommend **stochastic inference (test-time perturbation ensembling)** as the first uncertainty method to implement, for these reasons:
- Requires zero changes to `sen2sr` and zero additional trained artifacts — fully compatible with the "reuse unchanged" mandate.
- Compute cost is bounded and predictable (a fixed N× multiple of a single already-measured inference time — Baseline 0's 0.1553 s means even N=20 is ~3 s, tractable for an interactive demo).
- Conceptually complements, rather than duplicates, the existing LAM sensitivity tool (Section 14) — LAM already perturbs inputs via blur to study gradient sensitivity; test-time perturbation ensembling reuses a similar intuition (output stability under small input changes) to produce a genuine per-pixel *uncertainty* map rather than an attribution map.
- MC dropout is not recommended as the *first* method because its applicability is unverified (open question above) and its usefulness depends entirely on whether the pretrained architecture happens to have inference-relevant dropout — an assumption we should not build the plan around before checking.
- Deep ensembles are the strongest candidate scientifically but are gated on checkpoint availability that has not yet been confirmed; worth pursuing as a *second* method (or an upgrade path) once checkpoint availability is verified, not as the first deliverable.
- A learned uncertainty head is explicitly deferred — it depends on solving the training-data/ground-truth problem this document has just spent Section 11 being honest about not having solved.

This recommendation is a plan, not an implementation — Section 19 defines a dedicated "uncertainty experiment" as the concrete step where this gets built and its actual output (not a hypothesized one) gets evaluated.

---

## 14. LAM (Local Attribution Map)

**[FACT, restated for clarity]** LAM, as implemented in `sen2sr/xai/lam.py`, is strictly an **explainability/sensitivity** tool, not an uncertainty estimator:

- It answers: *"which input regions, and at which spatial frequency/blur scale, most influence this specific output patch?"* — computed via gradients of a windowed attribution objective with respect to progressively blurred copies of the input.
- It does **not** answer: *"how confident is the model in this output pixel's value?"* — it has no notion of a probability distribution over possible outputs, no variance, no calibration. Its outputs (a KDE saliency map, a Gini-based "complexity" scalar, a blur-robustness curve/scalar) describe *sensitivity to input perturbation*, which is a property of the model's gradient landscape, not a statement about correctness or confidence.

**How this differs from uncertainty (Section 13):** uncertainty (once implemented) will produce a signal answering "is this pixel likely wrong?" that a user should read as a *trust* signal per output pixel. LAM answers "why does the model behave this way here, and how robust is that behavior to blur?" — a *diagnostic/research* signal, most useful to someone inspecting model behavior, not to an end user deciding whether to trust a specific map.

**How it will be presented to users (proposal):** LAM will be surfaced in the UI (Section 16) under a clearly separate label — e.g., "Model sensitivity / explainability," never adjacent to or blended visually with the uncertainty layer — with explanatory copy stating explicitly that it is not a confidence or accuracy indicator. It is a secondary, optional, expert-facing view, not part of the primary "should I trust this pixel" workflow.

---

## 15. Downstream Applications

**[PROPOSAL]** — deliberately scoped as lightweight, honest demonstrations layered on top of the SR + uncertainty output, **not** three independent ML projects. Each demo uses classical/analytical methods (indices, thresholds, simple differencing) applied to the SR output, not new trained models, keeping scope bounded and keeping the "what does FRAME add" story honest (Section 24).

- **Agriculture.** Compute NDVI (and optionally a second simple index, e.g. NDWI) from the SR RGBN output at the 2.5 m pixel grid, and visually/statistically compare against the same index computed from the native 10 m input — the demonstration is "finer-grained vegetation-health visualization becomes possible," not a new crop-classification model. Overlay the uncertainty layer so low-confidence NDVI regions are visibly flagged, reinforcing the uncertainty-awareness theme rather than treating NDVI output as unconditionally trustworthy.
- **Urban mapping.** A simple built-up/impervious-surface proxy from the SR output (e.g., a threshold or edge-density heuristic on the higher-resolution bands, or a lightweight off-the-shelf classical segmentation such as edge detection/simple thresholding — not a trained deep segmentation model), demonstrating that finer pixel density reveals built structures (roads, building footprints) that are blurred together at native 10 m. Framed as "resolution improves visual/analytical separability of urban features," not as a validated urban land-cover product.
- **Disaster / change detection.** A before/after differencing demo: run SR on two dates of the same AOI, difference the results (or a derived index) to highlight change, with the uncertainty layer used to suppress/flag change signals in regions where either date's SR output is low-confidence (avoiding the classic false-positive-change trap of comparing two independently noisy super-resolved images). This directly showcases *why* uncertainty-awareness matters for a real use case, tying Sections 13/15 together.

Explicitly out of scope for this phase: training any new classifier, segmentation network, or change-detection model. All three demonstrations are analytical post-processing on top of the existing SR output, sized to be buildable in the time available and honest about being demonstrations, not validated operational products.

---

## 16. User Experience

**[PROPOSAL]** — not built. Target workflow for the eventual frontend:

1. **Upload / AOI.** User either draws/selects an AOI + date on a map (triggering the STAC ingestion path) or uploads their own georeferenced raster.
2. **Preview.** Show the native-resolution input (true-color composite) with basic metadata (date, CRS, bands available, cloud coverage if known) before committing to a run.
3. **Run SR.** User selects the SR variant (RGBN-only vs. full cascade, if both are exposed) and triggers processing; UI shows progress (tiling progress for large AOIs).
4. **Compare.** Side-by-side (or slider) comparison of the bicubic-upsampled input vs. the SR output, echoing the visualization Baseline 0 already produces (`comparison_side_by_side.png`) but interactive.
5. **Uncertainty.** A togglable overlay showing the uncertainty layer, with explanatory text distinguishing it from LAM (Section 14) and stating its limitations (Section 13) inline — not just in documentation the user has to seek out.
6. **Metrics.** Where applicable (i.e., not for arbitrary live-deployment requests without a reference — Section 11.C), show whichever no-reference/consistency metrics were computed (Section 9.2/12), each labeled with what it does and does not prove.
7. **Analysis.** Optional panel to run one of the Section 15 downstream demonstrations on the current SR output.
8. **Download.** Export the SR GeoTIFF, the uncertainty GeoTIFF, and a run-metadata JSON (mirroring Baseline 0's metadata philosophy) as a bundle.

Frontend implementation itself is explicitly **not** part of this phase (per the task instructions) — this section defines the target workflow only.

---

## 17. API Contract

**[PROPOSAL]** — not implemented. FastAPI is the assumed framework per the project's eventual direction, but no endpoint below exists yet.

| Endpoint | Purpose | Request → Response (conceptual) |
|---|---|---|
| `POST /aoi/preview` | Fetch and preview a Sentinel-2 scene for a given AOI/date without running SR | AOI (lat/lon or bbox) + date range → preview image URL/bytes + scene metadata (date, cloud %, CRS) |
| `POST /upload` | Accept a user-supplied georeferenced raster | Multipart file → validated ingestion result (accepted bands, CRS, resolution) or a rejection with a specific input-contract violation (Section 5) |
| `POST /sr/run` | Execute the full SR pipeline on a previously previewed/uploaded input | Ingestion reference + SR variant choice → job ID (async) or direct result for small AOIs |
| `GET /sr/status/{job_id}` | Poll job progress for large/tiled AOIs | Job ID → status (queued/running/done/failed) + progress info |
| `GET /sr/result/{job_id}` | Retrieve the finished SR output metadata and preview | Job ID → SR raster reference, uncertainty raster reference, run metadata, computed consistency/no-reference metrics |
| `GET /sr/download/{job_id}/{artifact}` | Download a specific output artifact | Job ID + artifact name (`sr.tif`, `uncertainty.tif`, `preview.png`, `metadata.json`) → file stream |
| `POST /analysis/run` | Run a Section 15 downstream demonstration on a finished SR result | Job ID + analysis type (agriculture/urban/disaster) → derived raster/index + preview |
| `GET /lam/run` (or as a param on `/sr/run`) | Compute the existing upstream LAM explainability output for a given input/result, clearly namespaced apart from uncertainty | Job ID (+ window/scale params matching `sen2sr.lam`'s existing signature) → KDE map, complexity metric, robustness vector |

Data flow through these endpoints follows Section 4 exactly — the API layer is a thin orchestration/serving surface over the pipeline, not a place where new pipeline logic lives.

**[OPEN QUESTION]** Sync vs. async job model for `/sr/run` — depends on how large a typical AOI turns out to be in practice once tiling overhead (Section 19 experiments) is measured; Baseline 0's single 128×128 patch (0.15 s) suggests small AOIs could be synchronous, but multi-tile AOIs likely need the async job pattern shown above.

---

## 18. Repository Architecture

**[PROPOSAL]** — target structure. `sen2sr/` (upstream) stays exactly as it is today; everything new lives in clearly separated top-level directories.

```
FRAME/
├── sen2sr/                     # UNTOUCHED upstream package (Section 2.3/2.4)
│   ├── __init__.py
│   ├── nonreference.py
│   ├── referencex2.py
│   ├── referencex4.py
│   ├── utils.py
│   ├── models/
│   └── xai/
│
├── experiments/                # existing Phase-0-style reproducible experiments
│   └── baseline/                # Baseline 0 (done — Section 2.2)
│       ├── run_baseline.py
│       ├── README.md
│       ├── outputs/
│       └── metadata/
│   # future: baseline_preprocessing/, baseline_geoexport/, validation/,
│   #         uncertainty/, downstream/, e2e_demo/  (Section 19)
│
├── frame/                      # NEW — our own installable package (not yet created)
│   ├── __init__.py
│   ├── ingestion/               # STAC/cubo fetch, upload handling
│   ├── preprocessing/           # band selection, normalization, masking, tiling glue
│   ├── geospatial/              # CRS/affine/bounds handling, GeoTIFF I/O
│   ├── consistency/             # Section 9.2 quantitative spectral checks
│   ├── uncertainty/             # Section 13 implementation, once built
│   ├── validation/              # Section 11 benchmark harness + metrics (Section 12)
│   ├── analysis/                # Section 15 downstream demonstrations
│   └── api/                     # FastAPI app (Section 17), once built
│
├── frontend/                    # NEW — not yet created (Section 16)
│
├── docs/
│   └── FRAME_TECHNICAL_SPEC.md  # this document
│
├── tests/                       # existing upstream test scaffold + new frame/ tests
├── pyproject.toml                # upstream package definition — left as-is
└── README.md                     # upstream package README — left as-is
```

Principle: `frame/` depends on `sen2sr` (imports it as a library), never the reverse, and never edits it. Every new capability in Sections 7–17 gets its own subpackage under `frame/` rather than being scattered into `sen2sr/` or `experiments/`.

---

## 19. Experiment Plan

**[PROPOSAL]**, sequenced to match Section 23's dependency order. Each experiment is a standalone, reproducible script/directory under `experiments/`, following the Baseline 0 pattern (fixed inputs, recorded metadata, no silent randomness).

1. **Baseline 0 — [DONE, Section 2.2].** Prove the unmodified upstream inference path works reproducibly. Already complete.
2. **Baseline with preprocessing.** Re-run Baseline 0's scene through FRAME's new preprocessing layer (cloud/mask handling, explicit metadata capture) instead of the ad hoc inline preprocessing in `run_baseline.py`, and confirm the SR output is bit-for-bit (or numerically equivalent within floating-point tolerance) identical to Baseline 0's — proving the new preprocessing layer is a faithful superset, not a silent behavior change.
3. **Baseline + geospatial export.** Wrap Baseline 0's run with the Section 10 geospatial module and produce a real GeoTIFF; verify CRS/affine/bounds correctness against the known AOI using `rasterio`/`gdalinfo` (or equivalent) as an independent check outside our own code.
4. **Validation experiment.** Implement and run the Section 11.A synthetic-degradation benchmark first (fully self-contained, no external data dependency), report the Section 12 metrics with their stated caveats; attempt Section 11.B only if a suitable external reference dataset is found (open question, Section 11.B).
5. **Uncertainty experiment.** Implement the recommended stochastic-inference method (Section 13) on top of Baseline 0's exact scene, and report what the resulting uncertainty map actually looks like on real output — including calling out any surprises or failure modes observed, not just a "it worked" summary.
6. **Downstream-task experiment.** Implement one of the three Section 15 demonstrations (agriculture/NDVI is the lowest-dependency starting point since it needs only the RGBN bands already used throughout) against a real SR output, with the uncertainty overlay from experiment 5.
7. **End-to-end demo.** Chain ingestion → preprocessing → SR → geospatial export → uncertainty → one downstream analysis into a single script mirroring the eventual API's `/sr/run` flow, as the concrete proof that the pieces compose before any frontend work begins.

---

## 20. Success Criteria

**[PROPOSAL]** — acceptance criteria stated as verifiable conditions, deliberately without invented numerical performance targets (no "PSNR > X" claim appears here, consistent with Sections 11–12).

- Every experiment in Section 19 runs deterministically and produces a recorded metadata artifact, following the Baseline 0 precedent — "reproducible" is itself a criterion, not just "produces output once."
- The geospatial-export experiment's output GeoTIFF is independently verifiable (via `rasterio`/`gdalinfo` or a GIS tool) to have correct CRS, affine transform, and bounds matching the known input AOI.
- The synthetic-degradation validation experiment reports all five reference-based metrics (Section 12) with their computed values and their stated caveats — success is "an honest, complete, caveated report exists," not "the numbers exceed some threshold."
- The uncertainty experiment produces a per-pixel uncertainty map that is co-registered (same CRS/affine/bounds/grid) with its corresponding SR output, and a written assessment of whether the map's high-uncertainty regions correspond to visually/qualitatively plausible trouble spots (e.g., cloud edges, high-frequency texture) — a qualitative sanity check, not a claimed calibration guarantee.
- At least one downstream-analysis demonstration runs against a real SR output and visibly incorporates the uncertainty layer (not just the raw SR output) in its presentation.
- The end-to-end experiment successfully chains every stage in Section 4 without manual intervention between stages.
- `sen2sr/` remains byte-for-byte unmodified throughout (checkable via `git diff` against the original package at any point).
- Every user-facing (or reviewer-facing) output — UI text, exported metadata, documentation — correctly distinguishes the 2.5 m pixel grid from a native 2.5 m observation (Section 1.4), and correctly labels uncertainty vs. LAM (Section 14) wherever both appear.

---

## 21. Risks

- **Technical.** Tiling/stitching correctness at AOI edges under the geospatial re-attachment design (Section 10) is non-trivial to get exactly right; an off-by-one-pixel affine error would silently misplace every output pixel while still "looking" fine visually.
- **Data.** STAC catalog availability/consistency (Baseline 0's own README already notes the catalog could change what a given date window returns); external high-resolution reference data for Section 11.B may not be freely available or appropriately licensed for the AOIs of interest.
- **Model.** The pretrained `sen2sr` weights are a fixed, external dependency (via Hugging Face) — no control over upstream changing, deprecating, or altering the hosted artifact; MC-dropout applicability (Section 13) is unverified and might turn out to be inapplicable, narrowing the uncertainty-method choice.
- **Scientific.** The core structural risk named throughout Section 11: no native 2.5 m ground truth exists, so any accuracy claim risks being over-interpreted by stakeholders unfamiliar with SR evaluation nuance, regardless of how carefully this document caveats it — communication discipline in the UI/report is itself a risk-mitigation task, not just a documentation nicety.
- **Licensing.** This repository's `LICENSE` file states CC0 1.0 Universal (a public-domain dedication, present since the initial commit) — permissive and compatible with reuse; the root `README.md`'s license badge previously and incorrectly said MIT, corrected during the Phase 9 audit to match the actual `LICENSE` file (see `docs/FINAL_SCIENTIFIC_AUDIT.md`). Any future decision to source an external VHR reference dataset (Section 11.B) or a different uncertainty-supporting checkpoint must independently verify that dataset's/checkpoint's license before use, since that determination has not been made for any specific dataset yet.
- **Deployment.** GPU availability in whatever environment ultimately serves the API is unconfirmed for anything beyond the Baseline 0 dev machine (RTX 3050 Laptop GPU, 100.5 MiB peak for a single 128×128 patch) — larger AOIs, N× uncertainty passes, and concurrent users all multiply memory/compute demand in ways not yet load-tested.
- **GPU/memory.** Stochastic-inference uncertainty (Section 13's recommendation) multiplies per-request inference cost by N; combined with `predict_large` tiling for large AOIs, worst-case compute cost has not yet been measured and could be a bottleneck for an interactive demo.

---

## 22. Scope Control

**[PROPOSAL]** — explicitly **not** building unless time clearly permits after the core pipeline (Sections 3–14) and at least one downstream demo (Section 15) are solid:

- Training or fine-tuning any new SR model, or a learned uncertainty head (Section 13's fourth candidate) — deferred pending the ground-truth/training-data problem being solved first.
- Any of the three Section 15 downstream applications beyond a single, lightweight demonstration each — no full crop-classification, no full building-footprint extraction, no trained change-detection model.
- The Section 11.B external-reference benchmark, if no suitable licensed dataset can be found within the available time — 11.A and 11.C remain the guaranteed-deliverable validation tiers.
- Deep-ensemble uncertainty (Section 13), unless checkpoint diversity is confirmed available — stochastic inference is the guaranteed-deliverable method.
- Multi-user auth, billing, or production-grade job queuing for the API (Section 17) — a single-user/demo-scale synchronous or simple-async job model is sufficient for this phase.
- Support for Sentinel-2 L1C input, or any sensor other than Sentinel-2 L2A.
- A production frontend design system — a functional demonstration UI covering the Section 16 workflow is the target, not a polished product.

---

## 23. Implementation Order

**[PROPOSAL]** — dependency-aware; frontend deliberately last, matching the task's explicit instruction.

1. **Preprocessing layer** (Section 7) — everything downstream needs correctly-shaped, correctly-masked, metadata-tagged input; builds directly on Baseline 0.
2. **Geospatial handling** (Section 10) — needed before any output can be honestly called a deliverable artifact (a GeoTIFF), and needed before validation experiments can meaningfully compare co-registered rasters.
3. **Spectral consistency checks** (Section 9.2) — a relatively small, self-contained addition once preprocessing/geospatial plumbing exists; produces early confidence signals useful for debugging everything built after it.
4. **Validation harness — synthetic-degradation tier first** (Section 11.A, 12) — establishes the metrics infrastructure and an honest first accuracy narrative before uncertainty work needs it as a comparison baseline.
5. **Uncertainty estimation** (Section 13, stochastic inference) — depends on preprocessing/geospatial plumbing being stable so the uncertainty raster can be correctly co-registered and exported like any other output.
6. **Validation harness — external-reference tier, if practical** (Section 11.B) — pursued in parallel with or after step 5, gated on data availability (Section 11.B/21 open question).
7. **Downstream analysis demo(s)** (Section 15) — depends on steps 1–5 being in place (needs SR output + uncertainty overlay to be meaningful).
8. **API layer** (Section 17) — wraps the now-working pipeline; building this earlier would mean building an interface to functionality that doesn't exist yet.
9. **Frontend** (Section 16) — deliberately last; depends on a working, callable API.
10. **End-to-end demo experiment** (Section 19, item 7) — a final integration pass validating the whole chain, ideally exercised through the same API the frontend will call.

---

## 24. SIH Novelty / Contribution

Stated plainly, without inflation:

- **We did not invent the Sentinel-2 super-resolution model.** `sen2sr`/SEN2SRLite, its neural architectures, its Fourier hard-constraint mechanism, its tiling utility, and its LAM explainability tool are entirely the work of the upstream ESAOpenSR project (Cesar Aybar, Julio Contreras, and collaborators), used here as a dependency, unmodified, under this repository's CC0 1.0 Universal `LICENSE` (Sections 2.3, 2.4, 22).
- **Wrapping an existing model is not, by itself, the novelty being claimed.** Simply calling `sen2sr` and displaying its output (which is essentially all Baseline 0 does) is a smoke test, not a contribution — this document says so explicitly (Section 2.2).
- **What FRAME actually contributes, concretely, per this specification:**
  1. A **geospatial-consistency layer** that does not exist upstream at all (Sections 2.1, 10) — turning bare tensors into GIS-usable, correctly-georeferenced GeoTIFFs.
  2. An **honest, tiered validation methodology** (Section 11) that explicitly confronts the structural absence of native 2.5 m Sentinel-2 ground truth, rather than presenting an unqualified accuracy number — this scientific framing, not any single metric value, is the contribution.
  3. A **quantitative, uncertainty-aware framing** of SR output (Section 13) — upstream provides no uncertainty signal whatsoever; adding one (and being explicit about what it can and cannot claim, Section 13's limitations column) directly targets the "uncertainty-aware" requirement in the problem statement.
  4. A **clear separation between explainability (LAM, upstream) and uncertainty (new, ours)** (Section 14) — a conceptual clarification that prevents a common and easy-to-make conflation in SR literature and product framing.
  5. **Lightweight, honestly-scoped downstream demonstrations** (Section 15) that connect SR + uncertainty to concrete field use cases, without overreaching into full independent ML products.
  6. An **application/platform layer** (ingestion, preprocessing orchestration, API, frontend — Sections 3–4, 7, 16–17) that makes the whole pipeline usable by someone who is not a remote-sensing ML researcher, which is not something a research package like `sen2sr` was designed to provide.
- **What we will not claim:** that the underlying SR quality, the Fourier consistency mechanism, or the base model architectures are our invention; that any uncertainty method we ship is a rigorously calibrated confidence interval; that any validation number in this project proves pixel-level accuracy at 2.5 m in the absence of the ground truth Section 11.D describes not existing.
