# FRAME — Final Scientific Audit (Phase 9)

This document is the authoritative, final statement of what FRAME does and
does not establish, written at the end of Phase 9 (final integration and
hardening) after running the complete pipeline end to end, auditing every
GeoTIFF's georeferencing independently, auditing the entire codebase for
terminology violations, and re-verifying reproducibility. It supersedes
any looser or more optimistic framing that may appear in earlier working
notes. Where this document and any other document disagree, this one is
correct.

**Read this before presenting FRAME to an SIH evaluator, and before
writing any slide, demo script, or summary that describes what FRAME
proves.**

---

## A. What FRAME actually contributes

FRAME is an orchestration, validation, and demonstration layer built
**around** an unmodified, frozen, third-party super-resolution model
(`sen2sr`/`SEN2SRLite`). FRAME's own original contributions are:

1. **A tested, typed preprocessing pipeline** (`frame.preprocessing`) that
   turns a raw Sentinel-2 RGBN array (either raw digital numbers or
   already-scaled reflectance — both explicitly supported, never silently
   assumed) into the exact tensor shape/scale the frozen model expects,
   with a validity mask and full provenance metadata.
2. **A geospatial layer** (`frame.geospatial`) that preserves and
   correctly derives CRS/affine-transform/bounds through the 4× upscaling
   step, writes/reads real GeoTIFFs, and — critically — refuses to
   silently fabricate geospatial context that isn't present (a missing
   CRS is a hard error, never a default).
3. **A self-consistency diagnostic layer** (`frame.consistency`) that
   checks the SR output against its own low-resolution input (downsample
   agreement, NDVI/B08:B04 ratio preservation) — an internal sanity check,
   explicitly not a ground-truth accuracy measurement.
4. **A model-stability uncertainty layer** (`frame.uncertainty`) that runs
   the frozen model N times under geometric test-time augmentation and
   reports the per-pixel dispersion as a **relative model-stability
   proxy** — a genuinely new signal this project adds on top of the
   frozen model, which by itself outputs no uncertainty at all.
5. **A downstream NDVI demonstration** (`frame.analysis`) showing the SR
   output is usable for a real vegetation-index workflow, with the
   uncertainty signal related to where the native/SR-derived NDVI
   comparison disagrees most.
6. **An external-reference validation harness** (`frame.validation`)
   against the independent `opensr-test` benchmark, run honestly (full
   `spot` subset, no cherry-picking, both the frozen model and a bicubic
   baseline reported side by side).
7. **A FastAPI backend and React frontend** (`frame/api/`, `frontend/`)
   that expose the entire pipeline as a working prototype, enforcing the
   terminology and scientific-caveat discipline structurally (tests that
   fail if a forbidden phrase appears in a live response) rather than by
   convention alone.
8. **This end-to-end integration** (`experiments/end_to_end/`, Phase 9):
   proof that all of the above compose into one coherent, reproducible
   system on a single real scene, not just as isolated per-phase
   experiments.

FRAME did **not** design, train, or fine-tune any neural network, and did
not invent Fourier hard-constraint SR, the Mamba/Swin/CNN backbones, or
the LAM explainability mechanism.

## B. What is inherited from SEN2SR

Everything that actually turns a low-resolution tensor into a
higher-resolution one:

- The `SEN2SRLite/NonReference_RGBN_x4` neural network weights and
  architecture (`sen2sr/models/opensr_baseline/`), trained and published
  by the upstream ESAOpenSR project (Cesar Aybar, Julio Contreras, and
  collaborators) — **used unmodified, never fine-tuned, never retrained**,
  loaded read-only via `mlstac` from the upstream Hugging Face manifest.
- The model's Fourier hard-constraint mechanism and tiling utility.
- The LAM explainability tool (`sen2sr/xai/lam.py`) — present upstream,
  **not used or exposed by any FRAME API endpoint or UI view in this
  project**, and never described as "uncertainty" anywhere in this
  codebase (verified structurally — see Section D of the terminology
  audit below).
- `sen2sr/` is licensed as stated in this repository's own `LICENSE` file
  (CC0 1.0 Universal — see the license/attribution correction recorded in
  `docs/FRAME_TECHNICAL_SPEC.md` Section 22 and the Phase 9 report) and
  was never modified by any phase of this project — verified explicitly
  in Phase 9 via `git diff -- sen2sr/` (empty).

## C. What the validation experiments establish

Three genuinely independent lines of evidence exist, each answering a
different, narrower question than "is the SR output accurate":

1. **Self-consistency** (`frame.consistency`, exercised on the Baseline 0
   scene and again in the Phase 9 end-to-end run): the SR output, when
   downsampled back to the native grid, agrees closely with its own LR
   input (downsample RMSE ≈ 0.0033, normalized RMSE ≈ 1.4%, on the Phase
   9 run). This establishes the model is **behaving coherently with its
   own input** — not injecting arbitrary structure. It says nothing about
   whether the *added* fine detail is physically correct.
2. **External reference benchmark** (`frame.validation`, Phase 4, the full
   `opensr-test` `spot` subset, 9 samples, an independent higher-resolution
   reference from a **different sensor** — SPOT, not Sentinel-2):
   PSNR 33.22 dB (vs. 33.12 dB for plain bicubic upsampling), SSIM 0.830
   (vs. 0.818 bicubic), RMSE 0.0258 (vs. 0.0261 bicubic), SAM 2.07°
   (bicubic 1.85°), ERGAS 3.68 (bicubic 3.77). The frozen model is
   comparable to — and on most metrics marginally better than — a naive
   bicubic upsample against this external reference, on 9 samples. This
   establishes the model does not *degrade* reconstruction fidelity
   relative to a trivial baseline on an independent benchmark. It does
   **not** establish sub-bicubic-level accuracy gains are large,
   statistically significant, or would hold on Sentinel-2-specific
   scenes (SPOT is a different sensor, different acquisition date,
   different atmosphere/illumination).
3. **Downstream utility** (`frame.analysis`, NDVI demonstration, Phase 6
   and re-verified in the Phase 9 end-to-end run): native and SR-derived
   NDVI agree closely on the common grid (mean absolute difference ≈
   0.005, RMSE ≈ 0.007, on the Phase 9 run) and the model-stability
   uncertainty correlates positively (r ≈ 0.33–0.47 across runs) with
   where that comparison disagrees most. This establishes the SR-derived
   product is usable for a real downstream index computation without
   producing wildly divergent vegetation statistics, and that the
   uncertainty signal is not noise — it carries some real relationship to
   where the two grids disagree.

## D. What they do NOT establish

- **None of the above proves the SR output is geometrically or
  radiometrically accurate at 2.5 m.** No native 2.5 m Sentinel-2
  observation exists anywhere, for any scene, because Sentinel-2 has
  never observed the ground at 2.5 m — this is a structural fact about
  the sensor, not a gap this project could close with more validation.
- Self-consistency (C.1) cannot detect a *systematic* hallucination the
  model reproduces confidently and consistently — a model that always
  invents the same wrong fine structure from a given LR input would still
  pass a self-consistency check with flying colors.
- The external benchmark (C.2) uses a **different sensor, acquisition
  date, and platform** than Sentinel-2. Cross-sensor confounds
  (atmospheric/illumination differences, residual co-registration error,
  temporal land-cover change between acquisitions) apply to every number
  reported. It is evidence of general reconstruction competence under a
  benchmark protocol, not a direct measurement of Sentinel-2-2.5m
  accuracy.
- The downstream NDVI agreement (C.3) is **not independent evidence** —
  the SR-derived NDVI is computed from a prediction *derived from* the
  same native 10 m observation the native NDVI is computed from. Close
  agreement partly reflects that shared origin, not an independent
  confirmation of physical correctness at 2.5 m.
- **9 samples is a small evaluation set.** The Phase 4 aggregate
  statistics (mean ± std over 9 `spot` samples) should be read as an
  indicative, not a statistically powerful, comparison against bicubic.
- **One scene** (Baseline 0, a peri-urban/agricultural mosaic near
  Valencia, Spain) has been exercised through the full pipeline in Phase
  9. Nothing in this project establishes the pipeline's behavior
  generalizes across cloud cover, other land-cover types, other seasons,
  or other geographies — only that it runs correctly and reproducibly on
  this one real scene.

## E. What uncertainty currently means

The uncertainty signal (`frame.uncertainty`, exposed as `relative
model-stability uncertainty` in the API and UI) is the per-pixel standard
deviation of the frozen model's predictions across a fixed, deterministic
set of geometric test-time augmentations (identity, horizontal flip,
vertical flip, and the three 90°-multiple rotations — 6 members total,
seeded, no dropout, no random noise injection).

**It measures:** how much the model's own output changes when the *same*
real input is re-framed and the output re-aligned. A pixel where the
model agrees with itself across every re-framing gets a low value; a
pixel where re-framing changes the prediction gets a high value.

**It explicitly does not mean:**
- A calibrated probability of error, or any kind of confidence interval.
  Nothing in this pipeline has been calibrated against a known error
  distribution — no native 2.5 m ground truth exists to calibrate
  against (Section D).
- A physically rigorous uncertainty bound in the sensor-noise or
  radiometric sense.
- LAM (`sen2sr/xai/lam.py`), which is a *different diagnostic entirely* —
  LAM answers "which input pixels most influence this output" via
  gradients on progressively blurred input copies (an explainability/
  sensitivity tool); this uncertainty signal answers "how much does the
  model disagree with itself across equivalent views of the same input"
  via repeated forward passes with no gradients (a stability proxy). They
  are never conflated anywhere in this codebase — enforced structurally
  by `frame/tests/test_api_terminology.py` and
  `frontend/src/__tests__/terminology.test.tsx`, and independently
  re-verified by grep across the whole repository during this Phase 9
  audit (zero violations found — see the Phase 9 report's terminology
  section).

**What it is legitimately useful for:** triage. "Look here first" — a
spatial map of where the model's own output is least self-consistent,
correlating with real image content (edges, fine structure, road/field
boundaries — visually confirmed in the Phase 9 summary visualization and
the frontend's uncertainty overlay), not with a fabricated or uniform
pattern.

## F. What the NDVI demonstration establishes

That FRAME's pipeline can support a genuine downstream remote-sensing
workflow (vegetation-index computation) end to end, at higher pixel
density, while explicitly exposing model-stability uncertainty alongside
the result — and that the resulting numbers are internally coherent (see
C.3) rather than nonsensical. It demonstrates **downstream utility and
internal consistency**, not proof that the SR-derived NDVI is a more
physically accurate measurement of vegetation vigor than the native 10 m
NDVI would be on its own.

## G. Current limitations

- Single validated scene for the full end-to-end pipeline (Baseline 0);
  single external benchmark subset (opensr-test `spot`, 9 samples) for
  reference-based validation.
- RGBN 4-band path only (B04/B03/B02/B08) — the other 6 Sentinel-2 bands
  (20 m red-edge/SWIR) are out of scope for every phase of this project.
- No cloud-cover, seasonal, or biome-diversity robustness testing.
- The API is a synchronous, single-process prototype with an in-memory
  job registry (`frame/api/README.md`) — not a production service:
  restarting the process discards all job/analysis records, there is no
  authentication, no persistent database, and no horizontal scaling.
- The frontend's AOI/date-window preview (`POST /aoi/preview`) validates
  request *shape* only — it does not fetch real imagery. There is no
  live STAC-driven "pick any scene" workflow in the shipped prototype.
- Bit-for-bit reproducibility (Phase 9, Section I below) has been
  demonstrated only on one machine/GPU/driver/PyTorch-build combination,
  run twice — it is not a claim about cross-hardware determinism.
- Model loading and the STAC-based scene-fetching approach used by prior
  phases' own experiments (`baseline`, `uncertainty`, `consistency`) was
  found during Phase 9 to be vulnerable to **catalog drift**: re-querying
  the identical AOI/date/band request today returns 3 scene entries where
  earlier phases observed and indexed only 1. Phase 9's own end-to-end
  experiment works around this by reusing the exact saved input tensor
  rather than re-fetching — but this means the *live* `/upload` +
  `/sr/run` flow (which takes a user-supplied file, not a STAC fetch) is
  unaffected, while any *future* experiment that re-runs `baseline`,
  `uncertainty`, or `consistency`'s own fetch step is not guaranteed to
  retrieve the identical historical scene those phases originally
  analyzed.

## H. What should NOT be claimed during the SIH presentation

Do not say, imply, or let a slide/demo suggest:

- That FRAME produces a **native 2.5 m Sentinel-2 image** or **true 2.5 m
  observation**. Say: **"SR-derived product — 2.5 m pixel grid."**
- That the SR output has been validated as **accurate at 2.5 m**, or that
  any metric in this project constitutes **2.5 m ground truth**. No such
  ground truth exists for Sentinel-2, anywhere, ever — this is a
  structural fact about the sensor, stated explicitly and repeatedly
  throughout this project's own documentation (`docs/FRAME_TECHNICAL_SPEC.md`
  Section 11.D).
- That the uncertainty output is a **calibrated confidence score**, a
  **probability of error**, or an **accuracy percentage**. It is a
  **relative model-stability proxy** — say that, and only that.
- That **LAM** (if ever mentioned) is the same thing as, or a substitute
  for, the uncertainty signal. They answer different questions via
  different mechanisms (Section E).
- That the Phase 4 external-reference benchmark constitutes **native
  Sentinel-2 ground truth**. It is an independent, different-sensor
  reference used for a standard SR-evaluation protocol — useful evidence,
  not a ground-truth accuracy proof.
- That NDVI agreement between native and SR-derived grids **proves
  physical accuracy**. It demonstrates internal consistency and
  downstream utility (Section F).
- That anything in this project has been tested across many scenes,
  seasons, or cloud conditions. It has been tested rigorously on **one**
  real deterministic scene end to end, plus a 9-sample external
  benchmark subset — say exactly that if asked.
- That the API/frontend is production-ready infrastructure. It is an
  explicit, documented prototype (Section G).

## I. Reproducibility status

**Actually demonstrated in Phase 9** (`experiments/end_to_end/`, see its
README for full detail): running the complete pipeline twice, with the
same fixed seed (42), same cached model weights, same device (CUDA), and
the same input tensor, produced:

- Every metadata field except timestamps, per-stage timings, and
  output-path strings **identical** between runs.
- All 5 output GeoTIFFs (`sr_mean.tif`, `uncertainty.tif`,
  `ndvi_native.tif`, `ndvi_sr.tif`, `ndvi_diff.tif`) **bit-for-bit
  numerically identical** (`numpy.array_equal`, including NaN positions;
  max/mean absolute difference exactly `0.0`) — not merely "within
  tolerance."

This is a real, measured result on this specific machine (NVIDIA GeForce
RTX 3050 Laptop GPU), this specific PyTorch/CUDA build, and this specific
frozen model with no dropout or batch-norm running-stat updates at
inference. **It is not a general claim of bit-identical determinism
across different hardware, drivers, or library versions** — floating-point
GPU accumulation order is a well-documented source of cross-hardware
non-determinism, and this project makes no claim about behavior on a
different machine.

The independent georeferencing audit (Phase 9) additionally confirmed:
zero pixel-grid/footprint drift across every output raster — the
uncertainty raster sits on exactly the same grid as the SR mean
prediction, native NDVI and the NDVI difference map sit on exactly the
native 10 m grid, and SR-derived NDVI sits on exactly the SR 2.5 m grid,
verified via independent `rasterio` reads, not assumed from the writer
code.

## J. Remaining risks

- **Catalog drift** (Section G) affects any *future* re-run of the
  `baseline`/`uncertainty`/`consistency` experiments' own STAC-fetch
  step — someone re-running those scripts today should expect to
  possibly retrieve a different scene than the one originally analyzed,
  unless they reuse the already-saved tensors as Phase 9 does.
- **Small-N validation.** Both the external benchmark (9 samples) and the
  end-to-end pipeline (1 scene) are small evidence bases. Any claim of
  general accuracy or robustness beyond what Sections C/D state precisely
  would overreach the actual evidence.
- **Single-GPU reproducibility.** The bit-identical result (Section I) has
  not been tested on CPU-only inference, a different GPU architecture, or
  a different CUDA/PyTorch version; minor floating-point divergence would
  not be surprising in those configurations and would not indicate a bug.
- **Prototype-grade backend.** The in-memory job registry, synchronous
  execution model, and lack of authentication (Section G) mean this
  system is not appropriate to expose beyond a controlled demo
  environment without further engineering — explicitly out of scope for
  this project's stated phases.
- **Terminology discipline depends on continued enforcement.** The
  structural tests (`test_api_terminology.py`,
  `frontend/src/__tests__/terminology.test.tsx`) are what actually
  prevent regression here, not just convention — any future change to
  `frame/api/schemas.py`'s constants or the frontend's
  `constants/terminology.ts` should keep those tests green.
- **License/attribution correction.** Phase 9 found and corrected a
  genuine discrepancy: the repository's `LICENSE` file states CC0 1.0
  Universal, while the root `README.md`'s badge and three statements in
  `docs/FRAME_TECHNICAL_SPEC.md` incorrectly said MIT. Both have been
  corrected to match the actual `LICENSE` file; attribution to the
  upstream ESAOpenSR project (Cesar Aybar, Julio Contreras, and
  collaborators) was already accurate and was not altered. Anyone citing
  this project's license in a presentation or paper should cite CC0 1.0
  Universal, not MIT.
