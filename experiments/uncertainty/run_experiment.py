"""Phase 5 experiment -- stochastic TTA uncertainty on the deterministic
Baseline 0 scene.

Goal
----
Run FRAME's proven, unmodified SEN2SRLite/NonReference_RGBN_x4 model N times
via geometric test-time augmentation (frame.uncertainty) on the EXACT same
deterministic scene Baseline 0 / Phase 1-4 all use, and characterize the
resulting per-pixel prediction dispersion as a relative model-stability
uncertainty proxy -- co-registered with the SR output via frame.geospatial
(Phase 2, reused unchanged).

This script does NOT modify sen2sr/, Baseline 0, or any Phase 1-4 experiment
artifact, and trains nothing.

SCIENTIFIC FRAMING -- read before the numbers below
-----------------------------------------------------
This uncertainty estimate measures prediction stability under the selected
test-time perturbations. It is a relative model-stability signal and is NOT
a calibrated probability of error, confidence interval, or physically
rigorous uncertainty bound. It is also NOT LAM (sen2sr/xai/lam.py): LAM is
explainability/sensitivity (which input pixels matter, via gradients on
blurred copies); this is a stability/uncertainty proxy (how much the
model's own prediction disagrees with itself across equivalent geometric
views, via repeated forward passes, no gradients). See
frame/uncertainty/README.md for the full distinction. Nothing here implies
the 2.5 m SR output is a native 2.5 m Sentinel-2 observation -- no such
observation exists (docs/FRAME_TECHNICAL_SPEC.md Section 1.4).

Stages
------
  1. Resolve compute device (mirrors Baseline 0).
  2. Reuse the SAME cached SEN2SRLite/NonReference_RGBN_x4 model artifact.
  3. Fetch the SAME deterministic Sentinel-2 L2A scene Baseline 0 uses.
  4. Capture geospatial metadata (Phase 2 style) BEFORE any tensor conversion.
  5. Run frame.preprocessing.preprocess_rgbn (Phase 1) with require_geospatial=True.
  6. Run the PRIMARY uncertainty ensemble (N=6, the full default geometric
     transform set) via frame.uncertainty.run_stochastic_uncertainty.
  7. Derive output geospatial metadata (frame.geospatial, Phase 2) and
     export the mean SR prediction AND the per-band + overall uncertainty
     as co-registered GeoTIFFs.
  8. Run an N-sweep (N = 1, 4, 8, 16) to empirically check whether the
     uncertainty conclusions change with ensemble size -- N > 6 necessarily
     repeats the 6 available geometric transforms cyclically (documented,
     not a bug -- see frame/uncertainty/README.md).
  9. Save visualizations and a comprehensive JSON metadata report.

Usage
-----
    <venv>/bin/python experiments/uncertainty/run_experiment.py
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from enum import Enum
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cubo
import matplotlib.pyplot as plt
import mlstac
import numpy as np
import torch

from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, write_geotiff
from frame.preprocessing import PreprocessedInput, RGBN_BANDS, preprocess_rgbn
from frame.uncertainty import DEFAULT_TRANSFORMS, UncertaintyResult, run_stochastic_uncertainty
from frame.uncertainty.statistics import normalize_for_visualization

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
#
# MUST match experiments/baseline/run_baseline.py exactly -- this is what
# makes it "the same deterministic scene." Duplicated rather than imported,
# keeping each experiment standalone (same convention as every prior phase).
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METADATA_DIR = BASE_DIR / "metadata"

AOI_LAT = 39.49152740347753
AOI_LON = -0.4308725142800361

SCENE_START_DATE = "2023-01-15"
SCENE_END_DATE = "2023-01-16"
SCENE_TIME_INDEX = 0

BANDS = ["B04", "B03", "B02", "B08"]
assert list(RGBN_BANDS) == BANDS, "frame.preprocessing.RGBN_BANDS drifted from Baseline 0's BANDS"

NATIVE_RESOLUTION_M = 10.0
EDGE_SIZE_PX = 128

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

PRIMARY_SEED = 42
N_SWEEP = (1, 4, 8, 16)


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
# Stage 3 -- deterministic Sentinel-2 L2A fetch (identical query to Baseline 0)
# ---------------------------------------------------------------------------

def fetch_scene():
    print(
        f"[scene] fetching {SCENE_START_DATE}..{SCENE_END_DATE} at "
        f"({AOI_LAT}, {AOI_LON}), bands={BANDS}, edge={EDGE_SIZE_PX}px"
    )
    da = cubo.create(
        lat=AOI_LAT,
        lon=AOI_LON,
        collection="sentinel-2-l2a",
        bands=BANDS,
        start_date=SCENE_START_DATE,
        end_date=SCENE_END_DATE,
        edge_size=EDGE_SIZE_PX,
        resolution=NATIVE_RESOLUTION_M,
    )
    n = da.sizes["time"]
    if n == 0:
        raise RuntimeError(
            "No Sentinel-2 scene found in the configured date window -- "
            "the STAC catalog may have changed. See experiments/baseline/README.md."
        )
    print(f"[scene] {n} catalog item(s) returned for this window; using index {SCENE_TIME_INDEX}")
    return da


# ---------------------------------------------------------------------------
# Stage 4 -- geospatial metadata extraction (same approach as Phase 1-4)
# ---------------------------------------------------------------------------

def extract_geospatial_metadata(da, resolution_m: float):
    epsg = da.attrs["epsg"]
    x = da["x"].values
    y = da["y"].values
    origin_x = float(x[0]) - resolution_m / 2.0
    origin_y = float(y[0]) + resolution_m / 2.0
    transform = (resolution_m, 0.0, origin_x, 0.0, -resolution_m, origin_y)
    height = len(y)
    width = len(x)
    bounds = (origin_x, origin_y - height * resolution_m, origin_x + width * resolution_m, origin_y)
    return {"crs": f"EPSG:{epsg}", "transform": transform, "bounds": bounds}


# ---------------------------------------------------------------------------
# Stage 8 -- N-sweep transform-list construction
# ---------------------------------------------------------------------------

def build_transform_list(n: int):
    """The first `n` transforms, cycling through DEFAULT_TRANSFORMS (6
    distinct geometric symmetries) when n > 6. Repeating a transform
    contributes a numerically identical, already-seen prediction -- zero
    NEW dispersion information by construction, reported as-is rather than
    hidden, per this phase's explicit instruction not to assume variance
    behaves in any particular way as N grows."""
    if n < 1:
        raise ValueError("n must be >= 1")
    base = list(DEFAULT_TRANSFORMS)
    return tuple(base[i % len(base)] for i in range(n))


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
        return None  # tensors are never embedded in the JSON report
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Visualizations -- Stage 9
# ---------------------------------------------------------------------------

def to_rgb(img_chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    rgb = np.stack([img_chw[0], img_chw[1], img_chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


UNCERTAINTY_LABEL = "Relative model-stability uncertainty (higher = less stable under TTA)"


def save_sr_and_uncertainty_maps(mean_rgb: np.ndarray, overall_std_map: np.ndarray, path: Path) -> None:
    normalized = normalize_for_visualization(torch.from_numpy(overall_std_map)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(mean_rgb)
    axes[0].set_title("SR mean prediction (RGB)", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(normalized, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title(UNCERTAINTY_LABEL, fontsize=10)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="normalized (2nd-98th percentile) -- display scaling only")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_overlay(mean_rgb: np.ndarray, overall_std_map: np.ndarray, path: Path) -> None:
    normalized = normalize_for_visualization(torch.from_numpy(overall_std_map)).numpy()
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    ax.imshow(mean_rgb)
    im = ax.imshow(normalized, cmap="magma", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"SR mean prediction with uncertainty overlay\n{UNCERTAINTY_LABEL}", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="normalized -- display scaling only")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_histogram(overall_std_map: np.ndarray, distribution, path: Path) -> None:
    flat = overall_std_map.flatten()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.hist(flat, bins=60, color="#5b7fae", edgecolor="none")
    for value, label, style in [
        (distribution.mean, "mean", "-"),
        (distribution.median, "median", "--"),
        (distribution.p90, "p90", ":"),
        (distribution.p95, "p95", "-."),
    ]:
        ax.axvline(value, linestyle=style, color="black", linewidth=1, label=f"{label}={value:.5f}")
    ax.set_xlabel("per-pixel std (reflectance units, band-averaged)")
    ax.set_ylabel("pixel count")
    ax.set_title("Distribution of per-pixel model-stability uncertainty\n(raw values -- no normalization applied)", fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_per_transform_disagreement(result: UncertaintyResult, path: Path) -> None:
    names = [d.transform_name for d in result.per_transform_disagreement]
    values = [d.mean_abs_deviation_from_ensemble_mean or 0.0 for d in result.per_transform_disagreement]
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.bar(names, values, color="#a8611b")
    ax.set_ylabel("mean |prediction - ensemble mean|")
    ax.set_title("Per-transform disagreement from the ensemble mean", fontsize=11)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_n_sweep(sweep_results, path: Path) -> None:
    ns = [r["n"] for r in sweep_results]
    scalars = [r["scalar_summary"] for r in sweep_results]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    ax.plot(ns, scalars, marker="o", color="#1f6f5c")
    ax.set_xlabel("N (ensemble size)")
    ax.set_ylabel("scalar_summary (mean per-pixel std)")
    ax.set_title("Does the uncertainty summary change with N?\n(N>6 cyclically repeats the 6 geometric transforms)", fontsize=10)
    ax.set_xticks(ns)
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
    device = get_device()
    print(f"[device] {device}")

    weight_dir = ensure_weights(WEIGHTS_CACHE_DIR)
    model = mlstac.load(str(weight_dir)).compiled_model(device=device)

    da = fetch_scene()
    raw = da[SCENE_TIME_INDEX].compute().to_numpy()
    input_geo = extract_geospatial_metadata(da, NATIVE_RESOLUTION_M)
    timestamp = str(da["time"].values[SCENE_TIME_INDEX])
    print(f"[scene] input geospatial metadata: {input_geo}")

    preprocessed: PreprocessedInput = preprocess_rgbn(
        raw,
        band_names=BANDS,
        input_scale="raw_digital_number",
        resolution_m=NATIVE_RESOLUTION_M,
        nodata_value=0.0,
        crs=input_geo["crs"],
        transform=input_geo["transform"],
        bounds=input_geo["bounds"],
        acquisition_timestamp=timestamp,
        require_geospatial=True,
    )
    X = preprocessed.tensor.to(device)
    print(f"[preprocess] input shape {tuple(X.shape)}, mask coverage {preprocessed.mask.coverage():.6f}")

    # ---- Stage 6: PRIMARY uncertainty ensemble (N=6, default geometric set) ----
    print(f"[uncertainty] running primary ensemble: N={len(DEFAULT_TRANSFORMS)}, seed={PRIMARY_SEED}")
    t0 = time.time()
    result = run_stochastic_uncertainty(
        model, X, transforms=DEFAULT_TRANSFORMS, seed=PRIMARY_SEED, band_names=RGBN_BANDS
    )
    primary_seconds = time.time() - t0
    print(
        f"[uncertainty] scalar_summary={result.scalar_summary:.6f}  "
        f"mean={result.overall_distribution.mean:.6f} median={result.overall_distribution.median:.6f} "
        f"p90={result.overall_distribution.p90:.6f} p95={result.overall_distribution.p95:.6f} "
        f"max={result.overall_distribution.max:.6f}"
    )
    for d in result.per_transform_disagreement:
        print(f"[uncertainty]   {d.transform_name}: disagreement={d.mean_abs_deviation_from_ensemble_mean:.6f} "
              f"time={d.inference_seconds:.4f}s")

    # ---- Stage 7: geospatial export ----
    output_metadata = derive_output_metadata(
        preprocessed.metadata, scale_factor=RGBN_SCALE_FACTOR, output_band_names=preprocessed.metadata.band_names
    )

    sr_path = OUTPUTS_DIR / "sr_mean.tif"
    write_geotiff(sr_path, result.mean_prediction.numpy(), output_metadata)
    print(f"[geoexport] wrote {sr_path}")

    overall_std_map = result.std_prediction.mean(dim=0).numpy()  # (H, W)
    uncertainty_band_names = tuple(f"{b}_std" for b in RGBN_BANDS) + ("overall_std",)
    uncertainty_array = np.concatenate(
        [result.std_prediction.numpy(), overall_std_map[None]], axis=0
    )  # (5, H, W): one band per uncertainty quantity
    uncertainty_metadata = dataclasses.replace(output_metadata, band_names=uncertainty_band_names, nodata_value=None)
    uncertainty_path = OUTPUTS_DIR / "uncertainty.tif"
    write_geotiff(uncertainty_path, uncertainty_array, uncertainty_metadata)
    print(f"[geoexport] wrote {uncertainty_path} (bands: {uncertainty_band_names})")

    torch.save(result.mean_prediction, OUTPUTS_DIR / "sr_mean_tensor.pt")
    torch.save(result.std_prediction, OUTPUTS_DIR / "sr_std_tensor.pt")

    # ---- Stage 9: visualizations ----
    mean_rgb = to_rgb(result.mean_prediction.numpy())
    save_sr_and_uncertainty_maps(mean_rgb, overall_std_map, OUTPUTS_DIR / "sr_and_uncertainty.png")
    save_overlay(mean_rgb, overall_std_map, OUTPUTS_DIR / "uncertainty_overlay.png")
    save_histogram(overall_std_map, result.overall_distribution, OUTPUTS_DIR / "uncertainty_histogram.png")
    save_per_transform_disagreement(result, OUTPUTS_DIR / "per_transform_disagreement.png")

    # ---- Stage 8: N-sweep ----
    print(f"[n-sweep] running N in {N_SWEEP}")
    sweep_results = []
    for n in N_SWEEP:
        transforms_n = build_transform_list(n)
        t0 = time.time()
        r_n = run_stochastic_uncertainty(
            model, X, transforms=transforms_n, seed=PRIMARY_SEED, band_names=RGBN_BANDS,
            keep_per_member_predictions=False,
        )
        seconds = time.time() - t0
        print(
            f"[n-sweep] N={n:2d}  scalar_summary={r_n.scalar_summary:.6f}  "
            f"p95={r_n.overall_distribution.p95:.6f}  max={r_n.overall_distribution.max:.6f}  "
            f"time={seconds:.2f}s"
        )
        sweep_results.append(
            {
                "n": n,
                "transform_names": list(r_n.transform_names),
                "scalar_summary": r_n.scalar_summary,
                "overall_distribution": to_jsonable(r_n.overall_distribution),
                "total_seconds": seconds,
            }
        )
    save_n_sweep(sweep_results, OUTPUTS_DIR / "n_sweep.png")

    run_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- metadata JSON ----
    report = {
        "experiment": "uncertainty (Phase 5) -- stochastic TTA model-stability proxy, NOT calibrated, NOT LAM",
        "scientific_framing": (
            "This uncertainty estimate measures prediction stability under the selected "
            "test-time perturbations. It is a relative model-stability signal and is not a "
            "calibrated probability of error, confidence interval, or physically rigorous "
            "uncertainty bound. It is distinct from LAM (sen2sr/xai/lam.py): LAM is "
            "explainability/sensitivity (gradient-based, answers 'which input pixels matter'); "
            "this TTA ensemble dispersion is a stability/uncertainty proxy (forward-pass-based, "
            "answers 'how much does the model disagree with itself across equivalent views'). "
            "This does not establish that the SR output equals a native 2.5m Sentinel-2 "
            "observation -- no such native Sentinel-2 measurement exists."
        ),
        "date_time_utc": run_start,
        "git_commit": git_commit(),
        "model": {
            "name": MODEL_NAME,
            "artifact_source": MODEL_MANIFEST_URL,
            "weights_cache_dir": str(weight_dir),
        },
        "input_scene_identity": {
            "coordinates": {"lat": AOI_LAT, "lon": AOI_LON},
            "scene_date_window": {"start": SCENE_START_DATE, "end": SCENE_END_DATE},
            "scene_time_index": SCENE_TIME_INDEX,
            "scene_timestamp": timestamp,
            "bands": BANDS,
            "input_resolution_m": NATIVE_RESOLUTION_M,
            "input_geospatial_metadata": input_geo,
        },
        "input_tensor_shape": list(X.shape),
        "output_tensor_shape": list(result.mean_prediction.shape),
        "primary_ensemble": {
            "n": result.n,
            "transform_names": list(result.transform_names),
            "seed": result.seed,
            "total_seconds": primary_seconds,
            "scalar_summary": result.scalar_summary,
            "scalar_summary_definition": result.scalar_summary_definition,
            "overall_distribution": to_jsonable(result.overall_distribution),
            "per_band_distribution": to_jsonable(result.per_band_distribution),
            "per_transform_disagreement": to_jsonable(result.per_transform_disagreement),
        },
        "statistics_definitions": {
            "mean_prediction": "arithmetic mean across the N (de-transformed) ensemble predictions, per pixel/band",
            "variance_prediction": "POPULATION variance (divide by N, not N-1) across the N ensemble predictions, per pixel/band",
            "std_prediction": "sqrt(variance_prediction)",
            "scalar_summary": result.scalar_summary_definition,
            "overall_distribution": "distribution (mean/median/std/p90/p95/min/max) of the band-averaged per-pixel std map",
            "per_band_distribution": "same distribution statistics, computed separately for each band's own std map",
            "visualization_normalization": "2nd-98th percentile clip and rescale to [0,1], DISPLAY ONLY -- never used in any statistic above",
        },
        "n_sweep": sweep_results,
        "output_geospatial_metadata": {
            "crs": output_metadata.crs,
            "transform": list(output_metadata.transform),
            "bounds": list(output_metadata.bounds),
            "resolution_m": output_metadata.resolution_m,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "band_names": list(output_metadata.band_names),
        },
        "uncertainty_raster_band_names": list(uncertainty_band_names),
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "peak_gpu_memory_mib": round(torch.cuda.max_memory_allocated(device) / (1024**2), 1) if device.type == "cuda" else None,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "timestamps": {"run_start_utc": run_start, "run_end_utc": run_end},
        "caveats": [
            "Relative, architecture-conditioned model-stability proxy -- not a calibrated probability of error.",
            "Not a confidence interval and not a physically rigorous uncertainty bound.",
            "Distinct from LAM (explainability/sensitivity); this is a stability/uncertainty proxy.",
            "The 2.5 m pixel grid is a statistical inference, not a native 2.5 m Sentinel-2 observation.",
            "N>6 in the N-sweep necessarily repeats the 6 default geometric transforms cyclically.",
            "No numeric 'high uncertainty' threshold is applied anywhere in this report.",
        ],
    }
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] outputs  -> {OUTPUTS_DIR}")
    print(f"[done] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
