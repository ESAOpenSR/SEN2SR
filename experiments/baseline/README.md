# Baseline 0 — SEN2SRLite `NonReference_RGBN_x4` smoke experiment

A reproducible re-run of the Phase 0 baseline check: does the unmodified,
upstream SEN2SR inference path actually produce a correct 4× super-resolved
output on a real Sentinel-2 L2A scene? This directory turns that one-off check
into a script that gives the **same scene, same preprocessing, same output
shape** every time it's run.

This is **not** an accuracy evaluation. There is no reference high-resolution
image here and no metric is computed — see "What this is / isn't" below.

## What this does not touch

`run_baseline.py` imports and calls the upstream `sen2sr`/`mlstac`/`cubo`
stack exactly as the project's own `README.md` documents. It does not import
any internal module from `sen2sr/`, does not subclass or wrap any model
class, and does not change any file under `sen2sr/`. It is a *consumer* of
the published package, not a modification of it.

## Layout

```
experiments/baseline/
  run_baseline.py     # the experiment script (see inline stage comments)
  README.md            # this file
  outputs/             # tensors + PNG visualizations (created by the script)
  metadata/            # run_metadata.json (created by the script)
```

## What "Baseline 0" is, precisely

| | |
|---|---|
| Model | SEN2SRLite `NonReference_RGBN_x4` |
| Bands | B04, B03, B02, B08 (10 m) |
| AOI | lat `39.49152740347753`, lon `-0.4308725142800361` (same point as the upstream README examples) |
| Scene | Sentinel-2 L2A acquisition of **2023-01-15** at that AOI — chosen in Phase 0 because it is cloud-free here; the Phase 0 run's default (README) example date landed on a cloud-covered scene |
| Patch size | 128×128 px (the model's native size — no tiling) |
| Scale | ×4 (10 m → 2.5 m) |

## Determinism

The upstream README's own examples pick a scene with `da[11]` against a
**full-year** date range, which (a) downloads far more imagery than needed
and (b) is only "deterministic" in the sense that catalog ordering happens
to be stable — it says nothing about which scene you'll actually get if the
catalog changes. This script instead queries a **single-day window**
(`2023-01-15` to `2023-01-16`) around the specific acquisition already known
to be clear, and always takes time index `0`. That query returns in ~2
seconds instead of the ~4–10 seconds (and much larger payload) needed to pull
a full year of timesteps just to scan them.

If the STAC catalog ever stops returning a scene for that exact day, the
script raises a clear error rather than silently substituting a different
one — re-derive a new clear date deterministically (e.g. re-run the Phase 0
cloud-scan approach once) and update `SCENE_START_DATE`/`SCENE_END_DATE` in
`run_baseline.py`.

## What this is / isn't

- **Is**: a repeatable smoke test that the documented inference path
  (weights download → scene fetch → preprocessing → forward pass) still
  works, on a fixed scene, with the real shapes/timings recorded.
- **Isn't**: an accuracy benchmark. No PSNR/SSIM/SAM/ERGAS or any other
  metric is computed here — the upstream repository doesn't implement any
  (see the earlier repository audit), and inventing one for this baseline
  would overstate what's being measured. Quantitative evaluation against a
  reference high-resolution image is future work (Phase 2).

## Running it

Reuses the isolated Python 3.11 environment created during Phase 0 (`uv venv
--python 3.11`), with `torch`, `torchvision`, `timm`, `einops`, `tqdm`,
`sen2sr`, `mlstac`, `cubo`, and `matplotlib` installed into it. If that
environment doesn't exist yet:

```bash
uv venv --python 3.11 sen2sr_venv
uv pip install --python sen2sr_venv/bin/python torch torchvision
uv pip install --python sen2sr_venv/bin/python timm einops tqdm sen2sr mlstac \
    "git+https://github.com/ESDS-Leipzig/cubo.git" matplotlib
```

Then, from the repository root:

```bash
<path-to-venv>/bin/python experiments/baseline/run_baseline.py
```

Requires internet access (Hugging Face for weights on first run — cached
afterward; a live STAC endpoint for the imagery on every run).

### Weight caching

Model weights are **not** stored inside this repository. They're cached
under `~/.cache/sen2sr_baseline/SEN2SRLite_RGBN/` by default (override with
the `SEN2SR_BASELINE_WEIGHTS_DIR` environment variable) so re-running the
script doesn't re-download ~4.5 MB of weights every time, and so the git
working tree never picks up binary model artifacts.

## Outputs produced by a run

`outputs/`
- `input_tensor.pt` — raw preprocessed LR input tensor, shape `(4, 128, 128)`
- `sr_tensor.pt` — raw model output tensor, shape `(4, 512, 512)`
- `lr_input_rgb.png` — LR input rendered as true-color RGB, native size
- `lr_bicubic_rgb.png` — LR input bicubic-upsampled to output size (display baseline only, not a model output)
- `sr_result_rgb.png` — SR output rendered as true-color RGB
- `comparison_side_by_side.png` — bicubic baseline vs. SR output, same extent

`metadata/`
- `run_metadata.json` — coordinates, scene date/timestep, bands, input/output
  resolution and shape, model name and artifact source, inference time, peak
  GPU memory, device/GPU name, Python version, PyTorch version
