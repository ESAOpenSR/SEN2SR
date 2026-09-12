"""Phase 4 experiment -- FRAME reference-based validation vs. opensr-test.

Goal
----
Run FRAME's proven, unmodified SEN2SRLite/NonReference_RGBN_x4 model against
real LR/HR pairs from the `opensr-test` benchmark's `spot` subset (the
smallest of our three supported subsets -- see frame/validation/README.md),
and report all three Phase 4 metric groups (A: standard reference metrics
bicubic-vs-SEN2SR, B: opensr-test's own metrics, C: Phase 3 self-consistency)
for every sample, kept explicitly separate.

This script does NOT modify sen2sr/, Baseline 0, or the Phase 1/2/3
experiments in any way, and does NOT train anything -- evaluation only.

What this proves
-----------------
That the real SEN2SRLite/NonReference_RGBN_x4 model, evaluated against a
genuine, independently-sourced higher-resolution reference (SPOT satellite
imagery, 2.5 m), outperforms the naive bicubic baseline on standard
reference-based image-quality metrics -- the first time in this project any
number here has been checked against real higher-resolution measured
imagery rather than only self-consistency.

What this does NOT prove
-------------------------
See frame.validation.report.SCIENTIFIC_FRAMING, reproduced in full in every
saved report: this does not establish that the SR output equals a native
2.5 m Sentinel-2 observation, because no such native Sentinel-2 measurement
exists. SPOT is a different sensor, acquisition date, and platform.

Stages
------
  1. Resolve compute device (mirrors Baseline 0).
  2. Reuse the SAME cached SEN2SRLite/NonReference_RGBN_x4 model artifact.
  3. Load the `spot` opensr-test subset (reuses the local cache from Phase
     3.5/4 research if present; downloads the ~187 MB pickle otherwise).
  4. For every sample in the subset: extract the 4-band RGBN LR/HR pair,
     compute the bicubic baseline, run the real model, run
     frame.validation.run_validation_sample (all three metric groups).
  5. Save a per-sample JSON (full nested detail), a tidy per-metric CSV
     (subset, sample_id, metric, bicubic_value, sen2sr_value, difference,
     relative_change), an aggregate summary, and visualizations for one
     representative sample.

Usage
-----
    <venv>/bin/python experiments/validation/run_experiment.py
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import platform
import sys
import time
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import mlstac
import numpy as np
import opensr_test
import torch

from frame.validation import (
    RGBN_BAND_NAMES,
    ValidationReport,
    bicubic_upsample,
    extract_sample,
    load_subset,
    run_validation_sample,
)

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METADATA_DIR = BASE_DIR / "metadata"

SUBSET = "spot"  # smallest supported subset (9 samples, ~187 MB) -- see
# frame/validation/README.md's Storage table for the exact measured sizes
# of all three supported subsets before this experiment downloaded anything.
DATASET_VERSION_TAG = "v3"

MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"
MODEL_MANIFEST_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)
WEIGHTS_CACHE_DIR = Path(
    os.environ.get(
        "SEN2SR_BASELINE_WEIGHTS_DIR",
        Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN",
    )
)

# The sample used for the representative visualizations (Stage 5c).
VISUALIZATION_SAMPLE_INDEX = 0


# ---------------------------------------------------------------------------
# Stage 1 -- compute device (mirrors Baseline 0)
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage 2 -- model weights (reuse Baseline 0's cache, do not re-download)
# ---------------------------------------------------------------------------

def ensure_weights(cache_dir: Path) -> Path:
    manifest_path = cache_dir / "mlm.json"
    if manifest_path.exists():
        print(f"[weights] using cached artifact at {cache_dir}")
        return cache_dir
    print(f"[weights] downloading {MODEL_MANIFEST_URL} -> {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    mlstac.download(file=MODEL_MANIFEST_URL, output_dir=str(cache_dir))
    return cache_dir


# ---------------------------------------------------------------------------
# JSON serialization for nested (Enum-carrying) dataclasses
# ---------------------------------------------------------------------------

def to_jsonable(obj):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return None  # tensors are never embedded in the JSON report
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Stage 5b -- tidy per-metric CSV rows (Phase 4 item 10's minimum schema)
# ---------------------------------------------------------------------------

def tidy_rows_for_report(report: ValidationReport) -> list:
    rows = []
    std = report.standard_reference_metrics
    for metric_name, comparison in std.comparisons.items():
        rows.append(
            {
                "subset": report.subset,
                "sample_id": report.roi_id or report.sample_index,
                "metric": metric_name,
                "bicubic_value": comparison.bicubic_value,
                "sen2sr_value": comparison.sen2sr_value,
                "difference": comparison.absolute_change,
                "relative_change": comparison.relative_change,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Stage 5c -- visualizations for one representative sample
# ---------------------------------------------------------------------------

def to_rgb(img_chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    rgb = np.stack([img_chw[0], img_chw[1], img_chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def save_lr_bicubic_hr_sr_comparison(lr, bicubic, hr, sr, path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, img, title in zip(
        axes,
        [lr.numpy(), bicubic.numpy(), hr.numpy(), sr.numpy()],
        ["LR input (native)", "Bicubic baseline", "HR reference (SPOT, HRharm)", "SEN2SR output"],
    ):
        ax.imshow(to_rgb(img))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("Reference-based validation -- opensr-test SPOT sample (not native Sentinel-2 ground truth)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_error_maps(bicubic, sr, hr, path: Path) -> None:
    bicubic_err = (bicubic.numpy().astype(np.float64) - hr.numpy().astype(np.float64))
    sr_err = (sr.numpy().astype(np.float64) - hr.numpy().astype(np.float64))
    bicubic_map = np.abs(bicubic_err).mean(axis=0)
    sr_map = np.abs(sr_err).mean(axis=0)
    vmax = max(bicubic_map.max(), sr_map.max())

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(bicubic_map, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title("|bicubic - HR| (mean over bands)", fontsize=11)
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sr_map, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title("|SEN2SR - HR| (mean over bands)", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle("Error vs. real SPOT HR reference (same color scale)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_opensr_test_visualizations(lr, sr, hr, out_dir: Path) -> None:
    """Uses opensr-test's OWN plotting methods (plot_summary, plot_tc) --
    a second, independent opensr_test.Metrics().compute() call is made here
    only because those plotting methods need the live Metrics object
    (frame.validation.compute_opensr_test_metrics returns a plain dataclass,
    not the object itself, to keep that module's return type serializable)."""
    metrics = opensr_test.Metrics()
    metrics.compute(lr=lr, sr=sr, hr=hr)

    fig, _ = metrics.plot_summary()
    fig.savefig(out_dir / "opensr_test_summary.png", dpi=150)
    plt.close(fig)

    fig2 = metrics.plot_tc()
    if isinstance(fig2, tuple):
        fig2 = fig2[0]
    fig2.savefig(out_dir / "opensr_test_hallucination_omission_improvement.png", dpi=150)
    plt.close(fig2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    run_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    device = get_device()
    print(f"[device] {device}")

    weight_dir = ensure_weights(WEIGHTS_CACHE_DIR)
    model = mlstac.load(str(weight_dir)).compiled_model(device=device)

    print(f"[dataset] loading opensr-test subset {SUBSET!r} (version={DATASET_VERSION_TAG})")
    t0 = time.time()
    loaded = load_subset(SUBSET, version=DATASET_VERSION_TAG)
    load_seconds = time.time() - t0
    n_samples = loaded["L2A"].shape[0]
    print(f"[dataset] loaded {n_samples} sample(s) in {load_seconds:.1f}s")

    all_reports = []
    all_rows = []
    per_sample_timing = []

    for sample_index in range(n_samples):
        sample = extract_sample(
            loaded, subset=SUBSET, sample_index=sample_index, dataset_version=opensr_test.__version__
        )

        X = sample.lr_reflectance.to(device)
        if X.device.type == "cuda":
            torch.cuda.synchronize()
        t_infer = time.time()
        with torch.no_grad():
            sr = model(X[None]).squeeze(0)
        if X.device.type == "cuda":
            torch.cuda.synchronize()
        inference_seconds = time.time() - t_infer
        sr = sr.cpu()

        report = run_validation_sample(sample, sr)
        all_reports.append(report)
        all_rows.extend(tidy_rows_for_report(report))
        per_sample_timing.append({"sample_id": sample.roi_id, "inference_seconds": inference_seconds})

        std = report.standard_reference_metrics
        print(
            f"[sample {sample_index}] roi={sample.roi_id} "
            f"PSNR bicubic={std.bicubic.psnr_db:.2f}dB sen2sr={std.sen2sr.psnr_db:.2f}dB | "
            f"SSIM bicubic={std.bicubic.ssim:.4f} sen2sr={std.sen2sr.ssim:.4f} | "
            f"RMSE bicubic={std.bicubic.rmse:.4f} sen2sr={std.sen2sr.rmse:.4f}"
        )

        if sample_index == VISUALIZATION_SAMPLE_INDEX:
            bicubic = bicubic_upsample(sample.lr_reflectance, sample.scale_factor)
            save_lr_bicubic_hr_sr_comparison(
                sample.lr_reflectance, bicubic, sample.hr_reflectance, sr, OUTPUTS_DIR / "lr_bicubic_hr_sr_comparison.png"
            )
            save_error_maps(bicubic, sr, sample.hr_reflectance, OUTPUTS_DIR / "error_maps.png")
            save_opensr_test_visualizations(sample.lr_reflectance, sr, sample.hr_reflectance, OUTPUTS_DIR)

    # ---- aggregate summary across all evaluated samples ----
    aggregate = {}
    for metric_name in all_reports[0].standard_reference_metrics.comparisons:
        bicubic_values = [
            r.standard_reference_metrics.comparisons[metric_name].bicubic_value for r in all_reports
        ]
        sen2sr_values = [
            r.standard_reference_metrics.comparisons[metric_name].sen2sr_value for r in all_reports
        ]
        bicubic_values = [v for v in bicubic_values if v is not None and np.isfinite(v)]
        sen2sr_values = [v for v in sen2sr_values if v is not None and np.isfinite(v)]
        aggregate[metric_name] = {
            "bicubic_mean": float(np.mean(bicubic_values)) if bicubic_values else None,
            "bicubic_std": float(np.std(bicubic_values)) if bicubic_values else None,
            "sen2sr_mean": float(np.mean(sen2sr_values)) if sen2sr_values else None,
            "sen2sr_std": float(np.std(sen2sr_values)) if sen2sr_values else None,
            "n_samples": len(sen2sr_values),
        }

    run_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- save the tidy CSV (Phase 4 item 10's minimum schema) ----
    csv_path = OUTPUTS_DIR / "per_metric_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subset", "sample_id", "metric", "bicubic_value", "sen2sr_value", "difference", "relative_change"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # ---- save the full per-sample JSON report ----
    full_report = {
        "experiment": "validation (Phase 4) -- reference-based benchmark vs. opensr-test, NOT native Sentinel-2 ground truth",
        "scientific_framing": all_reports[0].scientific_framing,
        "dataset": {
            "name": "opensr-test",
            "subset": SUBSET,
            "package_version": opensr_test.__version__,
            "dataset_format_version": DATASET_VERSION_TAG,
            "hf_dataset_repo": "isp-uv-es/opensr-test",
            "hf_dataset_repo_commit": "e4600b9c74a621adeec047e5f6cc7a2d70a58134",
            "github_repo_commit": "b42b1cba8a04b32341044f1f29474e5448499158",
            "n_samples_evaluated": n_samples,
            "load_seconds": round(load_seconds, 2),
        },
        "model": {
            "name": MODEL_NAME,
            "artifact_source": MODEL_MANIFEST_URL,
            "weights_cache_dir": str(weight_dir),
        },
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "opensr_test_version": opensr_test.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "timestamps": {"run_start_utc": run_start, "run_end_utc": run_end},
        "per_sample_timing": per_sample_timing,
        "aggregate": aggregate,
        "per_sample_reports": [to_jsonable(r) for r in all_reports],
    }
    json_path = METADATA_DIR / "run_metadata.json"
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[done] CSV      -> {csv_path}")
    print(f"[done] JSON     -> {json_path}")
    print(f"[done] outputs  -> {OUTPUTS_DIR}")
    print("\n[aggregate summary]")
    for metric_name, stats in aggregate.items():
        print(
            f"  {metric_name}: bicubic={stats['bicubic_mean']:.4f}±{stats['bicubic_std']:.4f}  "
            f"sen2sr={stats['sen2sr_mean']:.4f}±{stats['sen2sr_std']:.4f}  (n={stats['n_samples']})"
        )


if __name__ == "__main__":
    main()
