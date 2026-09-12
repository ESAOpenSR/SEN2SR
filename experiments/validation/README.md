# Reference-based validation — Phase 4 experiment

Runs FRAME's proven, **unmodified** `SEN2SRLite/NonReference_RGBN_x4` model
against real LR/HR pairs from the `opensr-test` benchmark's `spot` subset
(see `frame/validation/README.md` for the full package contract), producing
all three Phase 4 metric groups for every sample, kept explicitly separate.

This experiment does **not** modify `sen2sr/`, Baseline 0, or the Phase
1/2/3 experiments. Evaluation only — no training loop exists anywhere in
this repository.

## Scientific framing — read before the numbers below

> This benchmark evaluates reconstruction against an independently sourced
> higher-resolution reference dataset (opensr-test's SPOT subset — a
> different satellite, sensor, and acquisition date than Sentinel-2). It
> does NOT establish that the SR output equals a native 2.5 m Sentinel-2
> observation, because no such native Sentinel-2 measurement exists.

The full text (`frame.validation.report.SCIENTIFIC_FRAMING`) is embedded
verbatim in `metadata/run_metadata.json`.

## Exact dataset/version used

| | |
|---|---|
| Benchmark package | `opensr-test` **1.3.3** (PyPI, released 2025-03-21) |
| GitHub commit referenced during development | `b42b1cba8a04b32341044f1f29474e5448499158` |
| HF dataset repo | `isp-uv-es/opensr-test`, commit `e4600b9c74a621adeec047e5f6cc7a2d70a58134` |
| Dataset format version | `v3` (`opensr_test.load(..., version="v3")`) |
| Subset evaluated | `spot` — **all 9 samples** (the full subset; no sampling/truncation) |

## Storage — checked before downloading anything

Exact `Content-Length` from the real Hugging Face resolve URLs, measured via
`curl -I` **before** any `opensr_test.load()` call:

| Subset | Size |
|---|---|
| `spot` | 196,120,408 bytes ≈ 187 MB |
| `spain_crops` | 140,385,942 bytes ≈ 134 MB |
| `spain_urban` | 100,276,037 bytes ≈ 96 MB |

No sample-level/streaming access exists in `opensr_test`'s API — it
downloads one whole pickle per subset. At ~96–187 MB each, this posed no
storage risk (34 GB free on this machine at the time of the run), so the
full `spot` subset was evaluated rather than an artificially truncated
sample. `spain_crops`/`spain_urban` are not yet downloaded — extending this
experiment to them only requires changing the `SUBSET` constant.

## Compute — measured, not estimated

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB), same machine as Phases 0–3 |
| Inference time per sample | well under 1 s (see `metadata/run_metadata.json` → `per_sample_timing`) |
| Dataset load time | <1 s once cached locally (~187 MB, already downloaded during Phase 3.5 research) |

No GPU memory pressure observed — consistent with Baseline 0's own
~100 MiB-peak measurement for the same model at the same patch size.

## Result of the last run — bicubic vs. SEN2SR, aggregated over all 9 samples

| Metric | Bicubic (mean ± std) | SEN2SR (mean ± std) | Direction |
|---|---:|---:|---|
| PSNR (dB) | 33.12 ± 5.08 | 33.22 ± 5.10 | higher is better |
| SSIM | 0.818 ± 0.107 | 0.830 ± 0.099 | higher is better |
| RMSE | 0.0261 ± 0.0148 | 0.0258 ± 0.0147 | lower is better |
| SAM (°) | 1.85 ± 1.43 | 2.07 ± 1.63 | lower is better |
| ERGAS | 3.77 ± 2.54 | 3.68 ± 2.46 | lower is better |

**Read honestly, not as a uniform win.** SEN2SR outperforms bicubic on
PSNR, SSIM, RMSE, and ERGAS, all by a modest margin — but is measurably
*worse* than bicubic on SAM (spectral angle), both on average and on most
individual samples (see `outputs/per_metric_results.csv` for every sample's
values). This is reported as-is, not smoothed into a single "SR wins"
headline — the CRITICAL instruction for this phase is exactly to avoid
collapsing metrics into one invented score, and that discipline applies to
the *interpretation* of the results, not only their storage format. A
plausible reading (not confirmed further here) is that the model's added
spatial detail comes with a small spectral-shape cost the Fourier hard
constraint's low-frequency lock doesn't fully prevent — worth a dedicated
look in a later phase, not claimed as proven by this one experiment.

Full per-sample numbers for all 9 ROIs are in `outputs/per_metric_results.csv`
and `metadata/run_metadata.json`.

### Group (B) — opensr-test's own metrics (sample `ROI_0037`, for illustration)

```
reflectance:    0.0010   (LR-consistency, L1)
spectral:       0.182°   (LR-consistency, SAD)
spatial:        0.0      (phase-correlation misalignment)
synthesis:      0.0019   (harmonized-SR high-frequency detail)
hallucination:  0.035
omission:       0.915
improvement:    0.050
```

High omission / low hallucination here reads as the model behaving
conservatively on this sample — declining to invent detail it can't
support, rather than fabricating plausible-looking but wrong structure.
Not claimed as a general property beyond this one sample; see
`outputs/opensr_test_hallucination_omission_improvement.png`.

### Group (C) — Phase 3 self-consistency (unchanged, reused)

Computed identically to Phase 3 — needs no HR reference at all. Full values
in `metadata/run_metadata.json` under `phase3_self_consistency`.

## Layout

```
experiments/validation/
  run_experiment.py    # this experiment
  README.md             # this file
  outputs/
    lr_bicubic_hr_sr_comparison.png
    error_maps.png                                    # |bicubic-HR| vs |SEN2SR-HR|
    opensr_test_summary.png                            # opensr-test's own plot_summary()
    opensr_test_hallucination_omission_improvement.png # opensr-test's own plot_tc()
    per_metric_results.csv                             # tidy: subset, sample_id, metric, bicubic_value, sen2sr_value, difference, relative_change
  metadata/
    run_metadata.json    # full nested per-sample reports + aggregate + environment + timestamps
```

## Running it

Same environment as Phases 0–3 (`sen2sr_venv/`), plus `opensr-test==1.3.3`
(installed via `uv pip install --python sen2sr_venv/bin/python opensr-test`
— see `frame/validation/README.md`'s Dependencies section for exactly what
that pulls in).

```bash
sen2sr_venv/bin/python experiments/validation/run_experiment.py
```

Requires internet access on first run (Hugging Face for the `spot` pickle
and the model weights, both cached afterward — `~/.config/opensr_test/` and
`~/.cache/sen2sr_baseline/`, respectively). No GPU is required (falls back
to CPU), but a GPU is used automatically when available, as in every prior
phase.

To extend to `spain_crops`/`spain_urban`: change `SUBSET` in
`run_experiment.py` — no other code changes needed, since
`frame.validation.opensr_test_adapter` already supports all three.
