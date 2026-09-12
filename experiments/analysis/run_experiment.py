"""Phase 6 experiment -- NDVI downstream demonstration on the deterministic
Baseline 0 / Phase 5 scene.

Goal
----
Demonstrate that FRAME can provide a finer-grained vegetation-index
visualization (NDVI on the 2.5 m SR pixel grid) while explicitly exposing
Phase 5's model-stability uncertainty alongside it -- NOT to prove that
2.5 m NDVI is ground truth.

This script does NOT modify sen2sr/, Baseline 0, or any Phase 1-5 artifact,
and trains nothing. It also does NOT re-fetch the scene or re-run the
model: per this phase's explicit instruction to avoid unnecessary
inference, it reuses three already-saved tensors from prior phases:

  - experiments/baseline/outputs/input_tensor.pt      (Phase 0's native LR reflectance)
  - experiments/uncertainty/outputs/sr_mean_tensor.pt  (Phase 5's mean SR prediction)
  - experiments/uncertainty/outputs/sr_std_tensor.pt   (Phase 5's per-band uncertainty)

All three come from the EXACT same deterministic scene (same AOI, date
window, time index, bands) -- verified by cross-referencing each
experiment's own saved metadata JSON below, not assumed.

SCIENTIFIC FRAMING -- read before the numbers below
-----------------------------------------------------
1. SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid, not
   a directly observed native 2.5 m vegetation measurement.
2. The native 10 m NDVI and SR-derived NDVI are not independent ground
   truths -- the SR output was itself derived from the same underlying
   10 m observation.
3. Agreement or disagreement between them measures internal consistency
   and downstream utility, not proof of physical accuracy.
4. Uncertainty here is the Phase 5 relative model-stability proxy, not a
   calibrated probability of NDVI error.

Stages
------
  1. Load the three reused tensors and cross-check scene identity against
     their source experiments' own saved metadata.
  2. Run frame.analysis.run_ndvi_analysis (native NDVI, SR NDVI, common-grid
     comparison, uncertainty relation).
  3. Save visualizations.
  4. Save a comprehensive JSON metadata report.

Usage
-----
    <venv>/bin/python experiments/analysis/run_experiment.py
"""

from __future__ import annotations

import dataclasses
import json
import platform
import sys
import time
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from frame.analysis import NDVIAnalysisReport, run_ndvi_analysis
from frame.preprocessing import RGBN_BANDS
from frame.uncertainty.statistics import normalize_for_visualization

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration / reused artifact paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METADATA_DIR = BASE_DIR / "metadata"

REPO_ROOT = _REPO_ROOT
BASELINE0_DIR = REPO_ROOT / "experiments" / "baseline"
UNCERTAINTY_DIR = REPO_ROOT / "experiments" / "uncertainty"

LR_TENSOR_PATH = BASELINE0_DIR / "outputs" / "input_tensor.pt"
SR_MEAN_TENSOR_PATH = UNCERTAINTY_DIR / "outputs" / "sr_mean_tensor.pt"
SR_STD_TENSOR_PATH = UNCERTAINTY_DIR / "outputs" / "sr_std_tensor.pt"
BASELINE0_METADATA_PATH = BASELINE0_DIR / "metadata" / "run_metadata.json"
UNCERTAINTY_METADATA_PATH = UNCERTAINTY_DIR / "metadata" / "run_metadata.json"

SCALE_FACTOR = 4
NATIVE_RESOLUTION_M = 10.0


# ---------------------------------------------------------------------------
# JSON serialization for nested (Enum/tensor-carrying) dataclasses
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
        return None
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

NDVI_CMAP = "RdYlGn"  # standard remote-sensing NDVI convention: red=low, green=high vegetation vigor
SR_LABEL = "SR-derived NDVI — 2.5 m pixel grid"
NATIVE_LABEL = "Native 10 m NDVI"
UNCERTAINTY_LABEL = "Relative model-stability uncertainty (Phase 5 -- not a calibrated probability)"


def _masked(ndvi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, ndvi, np.nan)


def save_single_ndvi_map(ndvi: np.ndarray, mask: np.ndarray, title: str, path: Path, vmin=-1, vmax=1) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    im = ax.imshow(_masked(ndvi, mask), cmap=NDVI_CMAP, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="NDVI")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_downsampled_sr_vs_native(
    native_ndvi: np.ndarray, sr_downsampled: np.ndarray, mask: np.ndarray, path: Path
) -> None:
    vmin, vmax = -1, 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(_masked(native_ndvi, mask), cmap=NDVI_CMAP, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{NATIVE_LABEL} (10 m)", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(_masked(sr_downsampled, mask), cmap=NDVI_CMAP, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{SR_LABEL}\ndownsampled to the 10 m grid for comparison", fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="NDVI")
    fig.suptitle("Both maps on the SAME (native 10 m) grid -- never compared at mismatched resolutions", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_absolute_difference(diff: np.ndarray, mask: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    im = ax.imshow(_masked(diff, mask), cmap="magma", vmin=0)
    ax.set_title("|downsampled SR NDVI − native NDVI|\n(internal consistency, not ground-truth error)", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="|ΔNDVI|")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_sr_ndvi_uncertainty_overlay(sr_ndvi: np.ndarray, sr_mask: np.ndarray, uncertainty_sr_grid: np.ndarray, path: Path) -> None:
    normalized = normalize_for_visualization(torch.from_numpy(uncertainty_sr_grid)).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    ax.imshow(_masked(sr_ndvi, sr_mask), cmap=NDVI_CMAP, vmin=-1, vmax=1)
    im = ax.imshow(normalized, cmap="Greys", alpha=0.5, vmin=0, vmax=1)  # continuous overlay, no binary mask
    ax.set_title(f"{SR_LABEL}\nwith {UNCERTAINTY_LABEL}", fontsize=9)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="normalized uncertainty (2nd-98th pct) -- display scaling only")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_histogram(native_ndvi: np.ndarray, native_mask: np.ndarray, sr_downsampled: np.ndarray, common_mask: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(native_ndvi[native_mask].flatten(), bins=50, alpha=0.6, label=NATIVE_LABEL, color="#2f7d4f")
    ax.hist(sr_downsampled[common_mask].flatten(), bins=50, alpha=0.6, label=f"{SR_LABEL} (downsampled)", color="#a8611b")
    ax.set_xlabel("NDVI")
    ax.set_ylabel("pixel count")
    ax.set_title("Distribution of NDVI values — native vs. SR-derived (downsampled)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def to_rgb(img_chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    rgb = np.stack([img_chw[0], img_chw[1], img_chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def save_comprehensive_side_by_side(
    sr_mean_rgb: np.ndarray, native_ndvi: np.ndarray, native_mask: np.ndarray,
    sr_ndvi: np.ndarray, sr_mask: np.ndarray, diff: np.ndarray, common_mask: np.ndarray, path: Path,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    axes[0].imshow(sr_mean_rgb)
    axes[0].set_title("SR mean prediction (RGB)", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(_masked(native_ndvi, native_mask), cmap=NDVI_CMAP, vmin=-1, vmax=1)
    axes[1].set_title(NATIVE_LABEL, fontsize=10)
    axes[1].axis("off")

    axes[2].imshow(_masked(sr_ndvi, sr_mask), cmap=NDVI_CMAP, vmin=-1, vmax=1)
    axes[2].set_title(SR_LABEL, fontsize=10)
    axes[2].axis("off")

    im = axes[3].imshow(_masked(diff, common_mask), cmap="magma", vmin=0)
    axes[3].set_title("|ΔNDVI| (native grid)", fontsize=10)
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    fig.suptitle("FRAME NDVI downstream demonstration — not a claim of native 2.5 m ground truth", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    run_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.time()

    for path in (LR_TENSOR_PATH, SR_MEAN_TENSOR_PATH, SR_STD_TENSOR_PATH, BASELINE0_METADATA_PATH, UNCERTAINTY_METADATA_PATH):
        if not path.exists():
            raise RuntimeError(
                f"Required artifact not found: {path}. Run experiments/baseline/run_baseline.py and "
                "experiments/uncertainty/run_experiment.py first -- this experiment reuses their outputs "
                "rather than re-fetching the scene or re-running the model."
            )

    baseline_metadata = json.loads(BASELINE0_METADATA_PATH.read_text())
    uncertainty_metadata = json.loads(UNCERTAINTY_METADATA_PATH.read_text())

    # Cross-check scene identity across the two source experiments before trusting them.
    baseline_scene = (
        baseline_metadata["coordinates"], baseline_metadata["scene_date_window"], baseline_metadata["scene_time_index"]
    )
    uncertainty_scene = (
        uncertainty_metadata["input_scene_identity"]["coordinates"],
        uncertainty_metadata["input_scene_identity"]["scene_date_window"],
        uncertainty_metadata["input_scene_identity"]["scene_time_index"],
    )
    if baseline_scene != uncertainty_scene:
        raise RuntimeError(
            f"Scene identity mismatch between Baseline 0 ({baseline_scene}) and the Phase 5 "
            f"uncertainty experiment ({uncertainty_scene}) -- refusing to silently analyze mismatched scenes."
        )
    print(f"[scene] identity confirmed matching across Baseline 0 and Phase 5: {baseline_scene}")

    lr_reflectance = torch.load(LR_TENSOR_PATH, weights_only=True)
    sr_mean_prediction = torch.load(SR_MEAN_TENSOR_PATH, weights_only=True)
    sr_std_prediction = torch.load(SR_STD_TENSOR_PATH, weights_only=True)
    print(f"[load] lr={tuple(lr_reflectance.shape)} sr_mean={tuple(sr_mean_prediction.shape)} sr_std={tuple(sr_std_prediction.shape)}")

    # ---- Stage 2: NDVI analysis ----
    report: NDVIAnalysisReport = run_ndvi_analysis(
        lr_reflectance, sr_mean_prediction, sr_std_prediction,
        band_names=RGBN_BANDS, scale_factor=SCALE_FACTOR, native_resolution_m=NATIVE_RESOLUTION_M,
    )
    comparison = report.ndvi_comparison.comparison
    print(
        f"[ndvi] native valid={report.native_ndvi.valid_mask.sum().item()} "
        f"sr valid={report.sr_ndvi.valid_mask.sum().item()} "
        f"comparison valid={comparison.valid_pixel_count}"
    )
    print(
        f"[comparison] status={comparison.status.value} mean_abs_diff={comparison.mean_abs_difference} "
        f"rmse={comparison.rmse} max_abs_diff={comparison.max_abs_difference}"
    )
    uw = report.uncertainty_weighted_summary
    print(
        f"[uncertainty] status={uw.status.value} "
        f"correlation(uncertainty, |dNDVI|)={uw.correlation_uncertainty_vs_abs_diff} "
        f"weighted_mean_abs_diff={uw.uncertainty_weighted_mean_abs_diff} "
        f"unweighted_mean_abs_diff={uw.unweighted_mean_abs_diff}"
    )

    # ---- Stage 3: visualizations ----
    native_ndvi_np = report.native_ndvi.ndvi.numpy()
    native_mask_np = report.native_ndvi.valid_mask.numpy()
    sr_ndvi_np = report.sr_ndvi.ndvi.numpy()
    sr_mask_np = report.sr_ndvi.valid_mask.numpy()
    sr_downsampled_np = report.ndvi_comparison.sr_ndvi_downsampled.numpy()
    diff_np = report.ndvi_comparison.absolute_difference_map.numpy()
    common_mask_np = report.common_grid_mask

    save_single_ndvi_map(native_ndvi_np, native_mask_np, NATIVE_LABEL, OUTPUTS_DIR / "ndvi_native_10m.png")
    save_single_ndvi_map(sr_ndvi_np, sr_mask_np, SR_LABEL, OUTPUTS_DIR / "ndvi_sr_2_5m.png")
    save_downsampled_sr_vs_native(native_ndvi_np, sr_downsampled_np, common_mask_np, OUTPUTS_DIR / "ndvi_downsampled_sr_vs_native.png")
    save_absolute_difference(diff_np, common_mask_np, OUTPUTS_DIR / "ndvi_absolute_difference.png")
    save_sr_ndvi_uncertainty_overlay(sr_ndvi_np, sr_mask_np, report.uncertainty_overall_sr_grid.numpy(), OUTPUTS_DIR / "sr_ndvi_uncertainty_overlay.png")
    save_histogram(native_ndvi_np, native_mask_np, sr_downsampled_np, common_mask_np, OUTPUTS_DIR / "ndvi_histogram.png")

    sr_mean_rgb = to_rgb(sr_mean_prediction.numpy())
    save_comprehensive_side_by_side(
        sr_mean_rgb, native_ndvi_np, native_mask_np, sr_ndvi_np, sr_mask_np, diff_np, common_mask_np,
        OUTPUTS_DIR / "comparison_side_by_side.png",
    )

    runtime_seconds = time.time() - t0
    run_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- metadata JSON ----
    full_report = {
        "experiment": "analysis (Phase 6) -- NDVI downstream demonstration, NOT a claim of native 2.5m ground truth",
        "scientific_caveats": list(report.scientific_caveats),
        "reused_artifacts": {
            "lr_reflectance": str(LR_TENSOR_PATH.relative_to(REPO_ROOT)),
            "sr_mean_prediction": str(SR_MEAN_TENSOR_PATH.relative_to(REPO_ROOT)),
            "sr_std_prediction": str(SR_STD_TENSOR_PATH.relative_to(REPO_ROOT)),
            "note": "No model inference and no scene re-fetch were performed by this experiment.",
        },
        "scene_identity": {
            "coordinates": baseline_metadata["coordinates"],
            "scene_date_window": baseline_metadata["scene_date_window"],
            "scene_time_index": baseline_metadata["scene_time_index"],
            "scene_timestamp": baseline_metadata.get("scene_timestamp"),
            "band_order": list(RGBN_BANDS),
        },
        "native_resolution_m": NATIVE_RESOLUTION_M,
        "sr_resolution_m": NATIVE_RESOLUTION_M / SCALE_FACTOR,
        "scale_factor": SCALE_FACTOR,
        "ndvi_formula": report.metadata["ndvi_formula"],
        "resampling_method": report.metadata["resampling_method"],
        "valid_pixel_counts": {
            "native_ndvi": report.metadata["native_valid_pixel_count"],
            "sr_ndvi": report.metadata["sr_valid_pixel_count"],
            "comparison_common_grid": report.metadata["comparison_valid_pixel_count"],
        },
        "comparison_metrics": to_jsonable(comparison),
        "uncertainty_weighted_summary": to_jsonable(uw),
        "geospatial_metadata": {
            "native": uncertainty_metadata["input_scene_identity"]["input_geospatial_metadata"],
            "sr_output": uncertainty_metadata["output_geospatial_metadata"],
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
        },
        "timestamps": {"run_start_utc": run_start, "run_end_utc": run_end},
        "runtime_seconds": round(runtime_seconds, 4),
    }
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[done] outputs  -> {OUTPUTS_DIR}")
    print(f"[done] metadata -> {metadata_path}")
    print(f"[done] runtime  -> {runtime_seconds:.3f}s")


if __name__ == "__main__":
    main()
