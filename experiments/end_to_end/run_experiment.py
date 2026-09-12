"""Phase 9 experiment -- the complete FRAME pipeline, end to end, on the
SAME deterministic scene used throughout Phases 0-8.

Goal
----
Prove FRAME works as one coherent system -- not as isolated per-phase
experiments -- by running every real stage in sequence against a single
real input:

    input -> preprocessing -> frozen SEN2SR -> geospatial export
          -> uncertainty -> consistency diagnostics -> NDVI analysis
          -> final artifacts

This script imports and calls the existing, already-tested `frame.*`
modules directly (`frame.preprocessing`, `frame.geospatial`,
`frame.uncertainty`, `frame.consistency`, `frame.analysis`) -- the same
functions `frame/api/services/pipeline.py` orchestrates for the live API.
It does not reimplement or duplicate any Phase 1-8 science, does not
modify `sen2sr/`, and trains nothing.

Why this reuses the saved Baseline 0 tensor instead of re-fetching
-----------------------------------------------------------------
Phases 0/1/3/5 (`experiments/baseline`, `baseline_preprocessing`,
`consistency`, `uncertainty`) all re-query the same `cubo.create(...)`
AOI/date-window/band request each time, on the stated assumption that the
STAC catalog returns the same single scene for that historical window
every time. That assumption was checked while building this phase and
found FALSE: re-running the identical query today returns **3** time
entries for the same window, not the single entry every prior phase
observed and indexed as `SCENE_TIME_INDEX = 0`. The catalog has drifted
(almost certainly newly-reprocessed/ingested scenes for that historical
date), so a fresh fetch can no longer be trusted to return the same
scene at index 0 that Phases 0-8 used -- re-fetching here risks *silently
substituting a different scene*, which this phase's instructions
explicitly forbid.

The one way to guarantee the literal same scene is to reuse the exact
tensor Baseline 0 already fetched and saved:
`experiments/baseline/outputs/input_tensor.pt`. Its geospatial context
(CRS/transform/bounds) is cross-checked against
`experiments/baseline_geoexport/metadata/run_metadata.json`'s
`input_geospatial_metadata` (captured from the same original fetch,
before any tensor conversion) rather than re-derived.

Note on pixel scale: this saved tensor's values are already in the 0-1
reflectance range (verified: min ~0.068, max ~0.72), not raw digital
numbers -- `input_scale="reflectance"` is used accordingly. Passing
"raw_digital_number" here would silently re-divide already-scaled
reflectance and produce a near-zero, meaningless SR output with no error
raised (this exact failure mode was found and fixed in Phase 8's
frontend upload flow; the fix there was a UI toggle, the fix here is
simply choosing the argument that matches this tensor's real content).

Usage
-----
    <venv>/bin/python experiments/end_to_end/run_experiment.py [--outputs-dir DIR] [--metadata-dir DIR]

The optional directory overrides exist only so the Phase 9 reproducibility
check (see experiments/end_to_end/README.md) can run this script a second
time into a separate location without overwriting the canonical run.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from frame.analysis import run_ndvi_analysis
from frame.consistency import run_consistency_diagnostics
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, read_geotiff, write_geotiff
from frame.preprocessing import PROVEN_PATCH_SIZE, RGBN_BANDS, preprocess_rgbn
from frame.preprocessing.metadata import RasterMetadata
from frame.uncertainty import DEFAULT_TRANSFORMS, run_stochastic_uncertainty

BASE_DIR = Path(__file__).resolve().parent
BASELINE0_DIR = _REPO_ROOT / "experiments" / "baseline"
GEOEXPORT_DIR = _REPO_ROOT / "experiments" / "baseline_geoexport"
VALIDATION_DIR = _REPO_ROOT / "experiments" / "validation"

INPUT_TENSOR_PATH = BASELINE0_DIR / "outputs" / "input_tensor.pt"
BASELINE0_METADATA_PATH = BASELINE0_DIR / "metadata" / "run_metadata.json"
GEOEXPORT_METADATA_PATH = GEOEXPORT_DIR / "metadata" / "run_metadata.json"
VALIDATION_METADATA_PATH = VALIDATION_DIR / "metadata" / "run_metadata.json"

MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"
MODEL_MANIFEST_URL = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json"
WEIGHTS_CACHE_DIR = Path(
    __import__("os").environ.get("SEN2SR_BASELINE_WEIGHTS_DIR", str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN"))
)
UNCERTAINTY_SEED = 42
NATIVE_RESOLUTION_M = 10.0
INPUT_SCALE = "reflectance"  # see module docstring -- this tensor is already reflectance-scaled


def to_jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return None  # tensors are never embedded in metadata JSON -- shapes/stats are recorded explicitly instead
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_weights(cache_dir: Path) -> Path:
    manifest_path = cache_dir / "mlm.json"
    if manifest_path.exists():
        return cache_dir
    import mlstac

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    mlstac.download(file=MODEL_MANIFEST_URL, output_dir=str(cache_dir))
    return cache_dir


def path_for_metadata(path: Path) -> str:
    """Repo-relative path when the file lives inside the repo (the normal
    case); the raw absolute path otherwise -- e.g. when --outputs-dir points
    outside the repo, as the Phase 9 reproducibility check does so its
    second run never overwrites the canonical committed one."""
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def to_rgb(chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    rgb = np.stack([chw[0], chw[1], chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def save_summary_figure(
    native_rgb: np.ndarray,
    sr_rgb: np.ndarray,
    uncertainty_overlay_map: np.ndarray,
    native_ndvi: np.ndarray,
    native_mask: np.ndarray,
    sr_ndvi: np.ndarray,
    sr_mask: np.ndarray,
    diff: np.ndarray,
    common_mask: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(native_rgb)
    axes[0, 0].set_title("Native 10 m input (RGB)", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sr_rgb)
    axes[0, 1].set_title("SR-derived product — 2.5 m pixel grid (RGB)", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sr_rgb)
    im0 = axes[0, 2].imshow(uncertainty_overlay_map, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    axes[0, 2].set_title("Relative model-stability uncertainty\n(normalized for display only)", fontsize=9)
    axes[0, 2].axis("off")
    fig.colorbar(im0, ax=axes[0, 2], fraction=0.046)

    def masked(arr, mask):
        return np.where(mask, arr, np.nan)

    axes[1, 0].imshow(masked(native_ndvi, native_mask), cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 0].set_title("Native 10 m NDVI", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(masked(sr_ndvi, sr_mask), cmap="RdYlGn", vmin=-1, vmax=1)
    axes[1, 1].set_title("SR-derived NDVI — 2.5 m pixel grid", fontsize=10)
    axes[1, 1].axis("off")

    im1 = axes[1, 2].imshow(masked(diff, common_mask), cmap="magma", vmin=0)
    axes[1, 2].set_title("|ΔNDVI| (native comparison grid)\ninternal consistency, not ground truth", fontsize=9)
    axes[1, 2].axis("off")
    fig.colorbar(im1, ax=axes[1, 2], fraction=0.046)

    fig.suptitle(
        "FRAME end-to-end pipeline (Phase 9) -- SR-derived product is a learned inference on a 2.5 m pixel grid, "
        "not a native Sentinel-2 2.5 m observation",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument("--metadata-dir", type=Path, default=BASE_DIR / "metadata")
    args = parser.parse_args()

    outputs_dir: Path = args.outputs_dir
    metadata_dir: Path = args.metadata_dir
    outputs_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    for path in (INPUT_TENSOR_PATH, BASELINE0_METADATA_PATH, GEOEXPORT_METADATA_PATH):
        if not path.exists():
            raise RuntimeError(f"Required Phase 0-2 artifact not found: {path}. Run experiments/baseline and experiments/baseline_geoexport first.")

    baseline_metadata = json.loads(BASELINE0_METADATA_PATH.read_text())
    geoexport_metadata = json.loads(GEOEXPORT_METADATA_PATH.read_text())
    input_geo = geoexport_metadata["input_geospatial_metadata"]

    run_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    stage_timings: Dict[str, float] = {}
    t_total = time.time()

    # ---- Stage 0: device + model (real, cached weights -- not re-downloaded) ----
    device = get_device()
    print(f"[device] {device}")
    t0 = time.time()
    weight_dir = ensure_weights(WEIGHTS_CACHE_DIR)
    model = __import__("mlstac").load(str(weight_dir)).compiled_model(device=device)
    stage_timings["model_load_seconds"] = round(time.time() - t0, 4)
    print(f"[model] loaded in {stage_timings['model_load_seconds']:.3f}s")

    # ---- Stage 1: input (reused deterministic Baseline 0 tensor) ----
    raw = torch.load(INPUT_TENSOR_PATH, weights_only=True)
    print(f"[input] shape={tuple(raw.shape)} min={raw.min().item():.4f} max={raw.max().item():.4f} mean={raw.mean().item():.4f}")
    if raw.shape[-1] != PROVEN_PATCH_SIZE or raw.shape[-2] != PROVEN_PATCH_SIZE:
        raise RuntimeError(f"Expected the proven {PROVEN_PATCH_SIZE}x{PROVEN_PATCH_SIZE} patch, got {tuple(raw.shape)}.")

    # ---- Stage 2: preprocessing (frame.preprocessing, Phase 1, reused) ----
    t0 = time.time()
    preprocessed = preprocess_rgbn(
        raw.numpy(),
        band_names=list(RGBN_BANDS),
        input_scale=INPUT_SCALE,
        resolution_m=NATIVE_RESOLUTION_M,
        nodata_value=0.0,
        crs=input_geo["crs"],
        transform=tuple(input_geo["transform"]),
        bounds=tuple(input_geo["bounds"]),
        acquisition_timestamp=baseline_metadata.get("scene_timestamp"),
        require_geospatial=True,
    )
    stage_timings["preprocessing_seconds"] = round(time.time() - t0, 4)
    print(f"[preprocess] mask coverage={preprocessed.mask.coverage():.6f} in {stage_timings['preprocessing_seconds']:.4f}s")

    # ---- Stage 3+5: frozen SEN2SR + uncertainty (frame.uncertainty, Phase 5, reused) ----
    # One TTA ensemble call performs both the frozen-model inference stage
    # AND the uncertainty stage -- its mean prediction IS the SR output,
    # exactly the convention frame/api/services/pipeline.py already uses.
    X = preprocessed.tensor.to(device)
    t0 = time.time()
    uncertainty_result = run_stochastic_uncertainty(model, X, transforms=DEFAULT_TRANSFORMS, seed=UNCERTAINTY_SEED, band_names=RGBN_BANDS)
    stage_timings["sr_plus_uncertainty_seconds"] = round(time.time() - t0, 4)
    print(f"[sr+uncertainty] n={uncertainty_result.n} scalar_summary={uncertainty_result.scalar_summary:.3e} in {stage_timings['sr_plus_uncertainty_seconds']:.4f}s")

    mean_prediction = uncertainty_result.mean_prediction.cpu()
    std_prediction = uncertainty_result.std_prediction.cpu()

    # ---- Stage 4: geospatial export (frame.geospatial, Phase 2, reused) ----
    t0 = time.time()
    output_metadata = derive_output_metadata(preprocessed.metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=preprocessed.metadata.band_names)

    sr_path = outputs_dir / "sr_mean.tif"
    write_geotiff(sr_path, mean_prediction.numpy(), output_metadata)

    overall_std = std_prediction.mean(dim=0).numpy()
    uncertainty_band_names = tuple(f"{b}_std" for b in RGBN_BANDS) + ("overall_std",)
    uncertainty_array = np.concatenate([std_prediction.numpy(), overall_std[None]], axis=0)
    uncertainty_metadata = dataclasses.replace(output_metadata, band_names=uncertainty_band_names, nodata_value=None)
    uncertainty_path = outputs_dir / "uncertainty.tif"
    write_geotiff(uncertainty_path, uncertainty_array, uncertainty_metadata)
    stage_timings["geospatial_export_seconds"] = round(time.time() - t0, 4)
    print(f"[geoexport] sr={sr_path.name} uncertainty={uncertainty_path.name} in {stage_timings['geospatial_export_seconds']:.4f}s")

    # ---- Stage 6: consistency diagnostics (frame.consistency, Phase 3, reused) ----
    t0 = time.time()
    diagnostics = run_consistency_diagnostics(
        lr=preprocessed.tensor, sr=mean_prediction, mask=preprocessed.mask.array, band_names=RGBN_BANDS, scale_factor=RGBN_SCALE_FACTOR
    )
    stage_timings["consistency_seconds"] = round(time.time() - t0, 4)
    print(f"[consistency] downsample_rmse={diagnostics.downsample_consistency.overall.rmse} in {stage_timings['consistency_seconds']:.4f}s")

    # ---- Stage 7: NDVI analysis (frame.analysis, Phase 6, reused) ----
    t0 = time.time()
    ndvi_report = run_ndvi_analysis(
        preprocessed.tensor, mean_prediction, std_prediction, band_names=RGBN_BANDS, scale_factor=RGBN_SCALE_FACTOR, native_resolution_m=NATIVE_RESOLUTION_M
    )
    native_ndvi_metadata = dataclasses.replace(preprocessed.metadata, band_names=("NDVI",), nodata_value=float("nan"))
    sr_ndvi_metadata = dataclasses.replace(output_metadata, band_names=("NDVI",), nodata_value=float("nan"))

    native_ndvi_path = outputs_dir / "ndvi_native.tif"
    write_geotiff(native_ndvi_path, ndvi_report.native_ndvi.ndvi.numpy()[None], native_ndvi_metadata)
    sr_ndvi_path = outputs_dir / "ndvi_sr.tif"
    write_geotiff(sr_ndvi_path, ndvi_report.sr_ndvi.ndvi.numpy()[None], sr_ndvi_metadata)
    ndvi_diff_path = outputs_dir / "ndvi_diff.tif"
    write_geotiff(ndvi_diff_path, ndvi_report.ndvi_comparison.absolute_difference_map.numpy()[None], native_ndvi_metadata)
    stage_timings["ndvi_analysis_seconds"] = round(time.time() - t0, 4)
    comparison = ndvi_report.ndvi_comparison.comparison
    print(f"[ndvi] mean_abs_diff={comparison.mean_abs_difference} rmse={comparison.rmse} in {stage_timings['ndvi_analysis_seconds']:.4f}s")

    # ---- Stage 8: summary visualization ----
    t0 = time.time()
    from frame.uncertainty.statistics import normalize_for_visualization

    overlay_normalized = normalize_for_visualization(torch.from_numpy(overall_std)).numpy()
    native_rgb = to_rgb(preprocessed.tensor.numpy())
    sr_rgb = to_rgb(mean_prediction.numpy())
    summary_path = outputs_dir / "summary.png"
    save_summary_figure(
        native_rgb, sr_rgb, overlay_normalized,
        ndvi_report.native_ndvi.ndvi.numpy(), ndvi_report.native_ndvi.valid_mask.numpy(),
        ndvi_report.sr_ndvi.ndvi.numpy(), ndvi_report.sr_ndvi.valid_mask.numpy(),
        ndvi_report.ndvi_comparison.absolute_difference_map.numpy(), ndvi_report.common_grid_mask,
        summary_path,
    )
    stage_timings["visualization_seconds"] = round(time.time() - t0, 4)

    # ---- Independent re-read verification (never trust write_geotiff blindly) ----
    for name, path, expected_bands, expected_shape in (
        ("sr", sr_path, output_metadata.band_names, (len(RGBN_BANDS),) + tuple(mean_prediction.shape[1:])),
        ("uncertainty", uncertainty_path, uncertainty_band_names, uncertainty_array.shape),
        ("native_ndvi", native_ndvi_path, ("NDVI",), (1,) + tuple(ndvi_report.native_ndvi.ndvi.shape)),
        ("sr_ndvi", sr_ndvi_path, ("NDVI",), (1,) + tuple(ndvi_report.sr_ndvi.ndvi.shape)),
        ("ndvi_diff", ndvi_diff_path, ("NDVI",), (1,) + tuple(ndvi_report.ndvi_comparison.absolute_difference_map.shape)),
    ):
        read_array, read_meta = read_geotiff(path)
        assert read_array.shape == expected_shape, f"{name}: shape mismatch on reread {read_array.shape} != {expected_shape}"
        assert read_meta.band_names == tuple(expected_bands), f"{name}: band name mismatch on reread"
        print(f"[verify] {name}: reread OK, shape={read_array.shape}, bands={read_meta.band_names}, crs={read_meta.crs}")

    stage_timings["total_seconds"] = round(time.time() - t_total, 4)
    run_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- validation cross-reference (a DIFFERENT dataset/scene -- cited, not recomputed) ----
    validation_reference = None
    if VALIDATION_METADATA_PATH.exists():
        validation_metadata = json.loads(VALIDATION_METADATA_PATH.read_text())
        validation_reference = {
            "source": str(VALIDATION_METADATA_PATH.relative_to(_REPO_ROOT)),
            "note": (
                "Phase 4's opensr-test 'spot' benchmark evaluates the SAME frozen model against an "
                "INDEPENDENT, DIFFERENT scene/sensor (SPOT reference imagery) -- not this Baseline 0 "
                "scene. Cited here for cross-reference only; not recomputed and not directly comparable "
                "pixel-for-pixel to this run's own self-consistency numbers."
            ),
            "dataset": validation_metadata.get("dataset"),
            "aggregate": validation_metadata.get("aggregate"),
        }

    scientific_caveats = [
        "The SR output is a learned statistical inference resampled onto a 2.5 m pixel grid -- Sentinel-2's "
        "finest native band resolution is 10 m, and it has never observed the ground at 2.5 m.",
        "Uncertainty here is a relative, architecture-conditioned model-stability proxy from test-time "
        "perturbation ensembling. It is NOT a calibrated probability of error, NOT a confidence interval, "
        "and NOT the upstream LAM explainability tool (sen2sr/xai/lam.py).",
        "Self-consistency diagnostics compare the SR output against its own LR input only -- not a "
        "ground-truth accuracy check.",
        "NDVI agreement between the native and SR-derived grids measures internal consistency and "
        "downstream utility, not proof of physical accuracy -- both are derived from the same underlying "
        "10 m observation, not independent ground truths.",
        "The Phase 4 opensr-test benchmark (cited below, if available) evaluates against an external, "
        "higher-resolution reference from a different sensor/scene -- it is not native Sentinel-2 2.5 m "
        "ground truth for this scene.",
    ]

    full_report: Dict[str, Any] = {
        "experiment": "end_to_end (Phase 9) -- the complete FRAME pipeline on the deterministic Baseline 0 scene",
        "scientific_caveats": scientific_caveats,
        "git_commit": git_commit(),
        "input_identity": {
            "source_tensor": str(INPUT_TENSOR_PATH.relative_to(_REPO_ROOT)),
            "reason_reused_not_refetched": (
                "Re-querying the same cubo/STAC AOI+date+band request during Phase 9 development returned "
                "3 time entries where Phases 0-8 observed and indexed only 1 -- the catalog has drifted, so "
                "a fresh fetch can no longer be trusted to return the same scene at index 0. Reusing the "
                "exact saved tensor is the only way to guarantee the identical scene, per this phase's "
                "explicit instruction not to silently substitute another one."
            ),
            "coordinates": baseline_metadata["coordinates"],
            "scene_date_window": baseline_metadata["scene_date_window"],
            "scene_time_index": baseline_metadata["scene_time_index"],
            "scene_timestamp": baseline_metadata.get("scene_timestamp"),
            "band_order": list(RGBN_BANDS),
            "input_scale": INPUT_SCALE,
            "input_value_range": {"min": float(raw.min()), "max": float(raw.max()), "mean": float(raw.mean())},
            "input_geospatial_metadata": input_geo,
        },
        "model_identity": {"name": MODEL_NAME, "manifest_url": MODEL_MANIFEST_URL, "weights_cache_dir": str(weight_dir)},
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "uncertainty_configuration": {
            "seed": UNCERTAINTY_SEED,
            "n": uncertainty_result.n,
            "transform_names": list(uncertainty_result.transform_names),
            "scalar_summary": uncertainty_result.scalar_summary,
            "scalar_summary_definition": uncertainty_result.scalar_summary_definition,
            "overall_distribution": to_jsonable(uncertainty_result.overall_distribution),
        },
        "shapes": {"input_shape": list(preprocessed.tensor.shape), "output_shape": list(mean_prediction.shape)},
        "bands": list(RGBN_BANDS),
        "geospatial": {
            "crs": output_metadata.crs,
            "native_transform": list(preprocessed.metadata.transform),
            "native_bounds": list(preprocessed.metadata.bounds),
            "native_width_height": [preprocessed.metadata.width, preprocessed.metadata.height],
            "sr_transform": list(output_metadata.transform),
            "sr_bounds": list(output_metadata.bounds),
            "sr_width_height": [output_metadata.width, output_metadata.height],
            "scale_factor": RGBN_SCALE_FACTOR,
            "native_resolution_m": NATIVE_RESOLUTION_M,
            "sr_resolution_m": NATIVE_RESOLUTION_M / RGBN_SCALE_FACTOR,
        },
        "self_consistency": to_jsonable(diagnostics),
        "ndvi_analysis": {
            "ndvi_formula": ndvi_report.metadata["ndvi_formula"],
            "resampling_method": ndvi_report.metadata["resampling_method"],
            "valid_pixel_counts": {
                "native_ndvi": ndvi_report.metadata["native_valid_pixel_count"],
                "sr_ndvi": ndvi_report.metadata["sr_valid_pixel_count"],
                "comparison_common_grid": ndvi_report.metadata["comparison_valid_pixel_count"],
            },
            "comparison_metrics": to_jsonable(comparison),
            "uncertainty_weighted_summary": to_jsonable(ndvi_report.uncertainty_weighted_summary),
        },
        "validation_reference": validation_reference,
        "artifacts": {
            "sr_geotiff": path_for_metadata(sr_path),
            "uncertainty_geotiff": path_for_metadata(uncertainty_path),
            "native_ndvi_geotiff": path_for_metadata(native_ndvi_path),
            "sr_ndvi_geotiff": path_for_metadata(sr_ndvi_path),
            "ndvi_diff_geotiff": path_for_metadata(ndvi_diff_path),
            "summary_visualization": path_for_metadata(summary_path),
        },
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_build": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
        },
        "timestamps": {"run_start_utc": run_start, "run_end_utc": run_end},
        "stage_timings_seconds": stage_timings,
    }

    metadata_path = metadata_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(full_report, f, indent=2)

    print(f"\n[done] outputs  -> {outputs_dir}")
    print(f"[done] metadata -> {metadata_path}")
    print(f"[done] total runtime -> {stage_timings['total_seconds']:.3f}s")


if __name__ == "__main__":
    main()
