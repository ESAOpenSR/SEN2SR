"""Phase 3 experiment -- FRAME spectral/self-consistency diagnostics vs. Baseline 0.

Goal
----
Run `frame.consistency.run_consistency_diagnostics` against a real SR result
produced from the exact same deterministic Baseline 0 scene (same AOI, same
date window, same time index, same bands, same unmodified
SEN2SRLite/NonReference_RGBN_x4 model), preprocessed by the Phase 1
`frame.preprocessing` layer and geospatially described the same way Phase 2
(`frame.geospatial`) established, and report what the diagnostics actually
measure on this scene.

This script does NOT modify `experiments/baseline/`,
`experiments/baseline_preprocessing/`, or `experiments/baseline_geoexport/`
in any way.

What this proves
-----------------
That `frame.consistency`'s diagnostics run end-to-end against a real model
output and produce real numbers -- not that the SR output is "accurate".
There is still no reference high-resolution image anywhere in this
experiment (see docs/FRAME_TECHNICAL_SPEC.md Section 11). Every value saved
here is a SELF-consistency diagnostic: how well the SR output, reduced back
to its LR grid, agrees with the real LR observation it came from. NONE of
the numbers below are labeled "accuracy" anywhere in this script's output.

Stages
------
  1. Resolve compute device (mirrors Baseline 0 / Phase 1 / Phase 2).
  2. Reuse the SAME cached SEN2SRLite/NonReference_RGBN_x4 model artifact.
  3. Fetch the SAME deterministic Sentinel-2 L2A scene Baseline 0 uses.
  4. Capture geospatial metadata (Phase 2 style) BEFORE any tensor conversion.
  5. Run frame.preprocessing.preprocess_rgbn (Phase 1) with require_geospatial=True.
  6. Run the unmodified upstream model.
  7. Derive output geospatial metadata (frame.geospatial.derive_output_metadata,
     Phase 2) for traceability -- this experiment does not re-write a GeoTIFF
     (already proven by experiments/baseline_geoexport/); it only records
     where the output pixels are.
  8. Run frame.consistency.run_consistency_diagnostics (Phase 3, new).
  9. Save a JSON diagnostics report, per-band results, NDVI comparison,
     B08/B04 comparison, and diagnostic visualizations.

Usage
-----
    <venv>/bin/python experiments/consistency/run_experiment.py
"""

from __future__ import annotations

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

import cubo
import matplotlib.pyplot as plt
import mlstac
import numpy as np
import torch

from frame.consistency import run_consistency_diagnostics
from frame.consistency.downsample import downsample_to_lr_grid
from frame.consistency.spectral_ratios import compute_ndvi, compute_simple_ratio
from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata
from frame.preprocessing import PreprocessedInput, RGBN_BANDS, preprocess_rgbn

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
#
# These values MUST match experiments/baseline/run_baseline.py exactly --
# that is what makes this "the same deterministic scene." Duplicated rather
# than imported, keeping each experiment standalone (same convention as
# experiments/baseline_preprocessing/ and experiments/baseline_geoexport/).
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
# Stage 4 -- geospatial metadata extraction (same approach as Phase 1/Phase 2)
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
    minx = origin_x
    maxx = origin_x + width * resolution_m
    maxy = origin_y
    miny = origin_y - height * resolution_m
    bounds = (minx, miny, maxx, maxy)

    return {"crs": f"EPSG:{epsg}", "transform": transform, "bounds": bounds}


# ---------------------------------------------------------------------------
# JSON serialization for the (nested, Enum-carrying) diagnostics dataclasses
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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Visualizations -- Stage 9
# ---------------------------------------------------------------------------

def save_index_comparison(lr_index: np.ndarray, sr_index: np.ndarray, mask: np.ndarray, *, title: str, path: Path) -> None:
    """Two-panel LR-derived vs. downsampled-SR-derived index map, plus their
    difference -- all at the shared LR grid resolution, masked pixels shown
    as blank (NaN)."""
    lr_display = np.where(mask, lr_index, np.nan)
    sr_display = np.where(mask, sr_index, np.nan)
    diff_display = np.where(mask, sr_index - lr_index, np.nan)

    vmin = np.nanmin([np.nanmin(lr_display), np.nanmin(sr_display)])
    vmax = np.nanmax([np.nanmax(lr_display), np.nanmax(sr_display)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im0 = axes[0].imshow(lr_display, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"{title} -- LR-derived (native 10 m)")
    axes[0].axis("off")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sr_display, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    axes[1].set_title(f"{title} -- downsampled-SR-derived")
    axes[1].axis("off")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    max_abs_diff = np.nanmax(np.abs(diff_display)) if np.isfinite(diff_display).any() else 1.0
    im2 = axes[2].imshow(diff_display, cmap="coolwarm", vmin=-max_abs_diff, vmax=max_abs_diff)
    axes[2].set_title(f"{title} -- discrepancy (SR-derived minus LR-derived)")
    axes[2].axis("off")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Self-consistency diagnostic (NOT an accuracy comparison) -- {title}", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_spatial_discrepancy_map(lr_np: np.ndarray, sr_down_np: np.ndarray, mask_np: np.ndarray, path: Path) -> None:
    """Per-pixel mean-absolute-error across bands (downsample-consistency),
    spatial map -- where in the image the SR output disagrees most with the
    real LR observation it was derived from, after downsampling back."""
    abs_err_per_band = np.abs(sr_down_np.astype(np.float64) - lr_np.astype(np.float64))
    mean_abs_err = abs_err_per_band.mean(axis=0)  # (H, W)
    display = np.where(mask_np, mean_abs_err, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    im = ax.imshow(display, cmap="magma")
    ax.set_title(
        "Downsample-consistency discrepancy map\n(mean |downsampled-SR - LR| across bands; NOT an accuracy map)",
        fontsize=10,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, label="reflectance units")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"[device] {device}")

    weight_dir = ensure_weights(WEIGHTS_CACHE_DIR)

    da = fetch_scene()
    raw = da[SCENE_TIME_INDEX].compute().to_numpy()  # (bands, H, W), raw digital numbers
    input_geo = extract_geospatial_metadata(da, NATIVE_RESOLUTION_M)
    timestamp = str(da["time"].values[SCENE_TIME_INDEX])
    print(f"[scene] input geospatial metadata: {input_geo}")

    # Stage 5 -- FRAME preprocessing (Phase 1), geospatial context required.
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
    print(f"[preprocess] mask coverage: {preprocessed.mask.coverage():.6f}")

    # Stage 6 -- unmodified upstream model.
    model = mlstac.load(str(weight_dir)).compiled_model(device=device)
    X = preprocessed.tensor.to(device)
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        sr = model(X[None]).squeeze(0)
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.time() - t0
    print(f"[inference] {inference_seconds:.3f}s, output shape {tuple(sr.shape)}")

    # Stage 7 -- output geospatial metadata (Phase 2), for traceability only
    # (no GeoTIFF written here -- that is experiments/baseline_geoexport/'s scope).
    output_metadata = derive_output_metadata(
        preprocessed.metadata,
        scale_factor=RGBN_SCALE_FACTOR,
        output_band_names=preprocessed.metadata.band_names,
    )
    print(f"[geospatial] output resolution_m={output_metadata.resolution_m} bounds={output_metadata.bounds}")

    # Stage 8 -- Phase 3: consistency diagnostics.
    #
    # cross_tile is explicitly None: this scene is a single 128x128 patch
    # (Baseline 0's own EDGE_SIZE_PX) and never exercises sen2sr.predict_large's
    # multi-tile path -- see frame/consistency/tiles.py and
    # frame/consistency/README.md for why no real multi-tile FRAME run
    # exists yet to compute a cross-tile result from.
    diagnostics = run_consistency_diagnostics(
        lr=preprocessed.tensor,
        sr=sr.cpu(),
        mask=preprocessed.mask.array,
        band_names=RGBN_BANDS,
        scale_factor=RGBN_SCALE_FACTOR,
        cross_tile=None,
    )

    dc = diagnostics.downsample_consistency
    print(f"[consistency] downsample-consistency overall: status={dc.overall.status.value} "
          f"rmse={dc.overall.rmse} mean_abs_error={dc.overall.mean_abs_error}")
    for band, bd in dc.per_band.items():
        print(f"[consistency]   {band}: status={bd.status.value} rmse={bd.rmse} mean_abs_error={bd.mean_abs_error}")
    print(f"[consistency] NDVI comparison: status={diagnostics.ndvi_comparison.status.value} "
          f"mean_abs_discrepancy={diagnostics.ndvi_comparison.mean_abs_discrepancy}")
    print(f"[consistency] B08/B04 ratio comparison: status={diagnostics.b08_b04_ratio_comparison.status.value} "
          f"mean_abs_discrepancy={diagnostics.b08_b04_ratio_comparison.mean_abs_discrepancy}")
    print("[consistency] cross_tile: None (single-patch scene, no predict_large tiling triggered -- see README)")

    # Stage 9 -- visualizations.
    lr_np = preprocessed.tensor.cpu().numpy()
    sr_down_np = downsample_to_lr_grid(sr.cpu(), RGBN_SCALE_FACTOR).numpy()
    mask_np = preprocessed.mask.array

    red_i = list(RGBN_BANDS).index("B04")
    nir_i = list(RGBN_BANDS).index("B08")

    lr_ndvi, lr_ndvi_computable = compute_ndvi(lr_np[nir_i], lr_np[red_i])
    sr_ndvi, sr_ndvi_computable = compute_ndvi(sr_down_np[nir_i], sr_down_np[red_i])
    ndvi_mask = mask_np & lr_ndvi_computable & sr_ndvi_computable
    save_index_comparison(lr_ndvi, sr_ndvi, ndvi_mask, title="NDVI", path=OUTPUTS_DIR / "ndvi_comparison.png")

    lr_ratio, lr_ratio_computable = compute_simple_ratio(lr_np[nir_i], lr_np[red_i])
    sr_ratio, sr_ratio_computable = compute_simple_ratio(sr_down_np[nir_i], sr_down_np[red_i])
    ratio_mask = mask_np & lr_ratio_computable & sr_ratio_computable
    save_index_comparison(
        lr_ratio, sr_ratio, ratio_mask, title="B08/B04 ratio", path=OUTPUTS_DIR / "b08_b04_ratio_comparison.png"
    )

    save_spatial_discrepancy_map(lr_np, sr_down_np, mask_np, OUTPUTS_DIR / "spatial_discrepancy_map.png")

    # Stage 9 (continued) -- save the JSON diagnostics report.
    report = {
        "experiment": "consistency (Phase 3) -- spectral/self-consistency diagnostics, NOT an accuracy benchmark",
        "model_name": MODEL_NAME,
        "model_artifact_source": MODEL_MANIFEST_URL,
        "model_weights_cache_dir": str(weight_dir),
        "coordinates": {"lat": AOI_LAT, "lon": AOI_LON},
        "scene_date_window": {"start": SCENE_START_DATE, "end": SCENE_END_DATE},
        "scene_time_index": SCENE_TIME_INDEX,
        "scene_timestamp": timestamp,
        "bands": BANDS,
        "input_resolution_m": NATIVE_RESOLUTION_M,
        "input_geospatial_metadata": input_geo,
        "output_geospatial_metadata": {
            "crs": output_metadata.crs,
            "resolution_m": output_metadata.resolution_m,
            "bounds": list(output_metadata.bounds),
            "width": output_metadata.width,
            "height": output_metadata.height,
        },
        "preprocessing_mask_coverage": preprocessed.mask.coverage(),
        "output_shape": list(sr.shape),
        "inference_seconds": round(inference_seconds, 4),
        "consistency_diagnostics": to_jsonable(diagnostics),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[done] outputs  -> {OUTPUTS_DIR}")
    print(f"[done] metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
