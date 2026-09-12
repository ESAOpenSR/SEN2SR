"""Phase 2 experiment -- FRAME geospatial export vs. Baseline 0.

Goal
----
Prove that `frame.geospatial` correctly preserves and reconstructs
geospatial context (CRS, affine transform, bounds) around the unmodified
upstream `sen2sr` SR pipeline, by running the exact same deterministic
Baseline 0 scene end to end -- fetch, FRAME preprocessing, unmodified
model, geospatial export, then independently reopening the written
GeoTIFF and verifying it matches what was intended.

This script does NOT modify `experiments/baseline/` or
`experiments/baseline_preprocessing/` in any way.

What this proves
-----------------
1. The geospatial metadata captured from the real fetched scene (before
   any tensor conversion) survives FRAME's preprocessing step unchanged.
2. `frame.geospatial.derive_output_metadata` produces an output transform
   whose recomputed footprint exactly matches the input footprint, and
   whose pixel size is exactly the input pixel size divided by the model's
   scale factor (4, for this path).
3. The GeoTIFF written from the SR tensor, when reopened independently via
   `frame.geospatial.read_geotiff`, reports the same CRS, transform,
   bounds, dimensions, resolution, and band order that were written --
   i.e. the round trip through an actual file on disk is lossless for
   geospatial context.

What this does NOT prove
-------------------------
Nothing about SR accuracy or quality -- see
docs/FRAME_TECHNICAL_SPEC.md Section 11. This experiment is scoped
exclusively to geospatial-layer correctness.

Stages
------
  1. Resolve compute device (mirrors Baseline 0 / Phase 1 experiment).
  2. Reuse the SAME cached SEN2SRLite/NonReference_RGBN_x4 model artifact.
  3. Fetch the SAME deterministic Sentinel-2 L2A scene Baseline 0 uses.
  4. Capture the scene's geospatial metadata BEFORE any tensor conversion.
  5. Run frame.preprocessing.preprocess_rgbn with require_geospatial=True.
  6. Run the unmodified upstream model.
  7. Derive output geospatial metadata (frame.geospatial.derive_output_metadata)
     and export the SR result as a GeoTIFF (frame.geospatial.write_geotiff).
  8. Reopen the GeoTIFF independently (frame.geospatial.read_geotiff) and
     verify CRS / transform / bounds / dimensions / resolution / band order.
  9. Explicitly check: input footprint == output footprint, and
     output pixel size == input pixel size / 4.
  10. Save a metadata report and an RGB preview rendered from the
      REOPENED file (not the in-memory tensor), so the preview itself is
      evidence the file round-trips correctly.

Usage
-----
    <venv>/bin/python experiments/baseline_geoexport/run_experiment.py
"""

from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cubo
import matplotlib.pyplot as plt
import mlstac
import numpy as np
import torch

from frame.geospatial import RGBN_SCALE_FACTOR, derive_output_metadata, read_geotiff, write_geotiff
from frame.preprocessing import PreprocessedInput, RGBN_BANDS, preprocess_rgbn

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
#
# These values MUST match experiments/baseline/run_baseline.py exactly --
# that is what makes this "the same deterministic scene."
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

# Floating-point-only tolerance for the invariant checks below (not a
# scientific threshold -- see frame/geospatial/transform.py's docstring).
FOOTPRINT_TOLERANCE_M = 1e-6
PIXEL_SIZE_TOLERANCE_M = 1e-9


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
# Stage 4 -- geospatial metadata extraction (captured BEFORE tensor conversion)
#
# cubo.create() deliberately deletes stackstac's own "crs"/"transform"
# attrs (see cubo/cubo.py) and only keeps `epsg` and `resolution`. The
# affine transform and bounds are recomputed here from those plus the
# cube's own pixel-center x/y coordinate arrays -- standard north-up raster
# geometry, not a guess. Identical logic to
# experiments/baseline_preprocessing/run_experiment.py's helper of the same
# name (duplicated rather than imported, keeping each experiment standalone).
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
# Stage 10 -- RGB preview from the REOPENED GeoTIFF
# ---------------------------------------------------------------------------

def to_rgb(img_chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    rgb = np.stack([img_chw[0], img_chw[1], img_chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def save_preview(array_from_file: np.ndarray, path: Path) -> None:
    rgb = to_rgb(array_from_file)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("SR output re-read from GeoTIFF (RGB preview)")
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

    # Stage 5 -- FRAME preprocessing, with geospatial context REQUIRED.
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
    print(f"[preprocess] input metadata: crs={preprocessed.metadata.crs} "
          f"transform={preprocessed.metadata.transform} bounds={preprocessed.metadata.bounds}")

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

    # Stage 7 -- derive output geospatial metadata and export as GeoTIFF.
    output_metadata = derive_output_metadata(
        preprocessed.metadata,
        scale_factor=RGBN_SCALE_FACTOR,
        output_band_names=preprocessed.metadata.band_names,
    )
    print(f"[geoexport] output metadata: crs={output_metadata.crs} "
          f"transform={output_metadata.transform} bounds={output_metadata.bounds} "
          f"resolution_m={output_metadata.resolution_m}")

    geotiff_path = OUTPUTS_DIR / "sr_output.tif"
    write_geotiff(geotiff_path, sr.cpu().numpy(), output_metadata)
    print(f"[geoexport] wrote {geotiff_path}")

    # Stage 8 -- reopen the GeoTIFF INDEPENDENTLY and verify.
    read_array, read_metadata = read_geotiff(geotiff_path)

    def isclose_seq(a, b, tol):
        return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))

    checks = {
        "crs_matches": read_metadata.crs == output_metadata.crs,
        "transform_matches": isclose_seq(read_metadata.transform, output_metadata.transform, 1e-6),
        "bounds_matches": isclose_seq(read_metadata.bounds, output_metadata.bounds, 1e-6),
        "width_matches": read_metadata.width == output_metadata.width,
        "height_matches": read_metadata.height == output_metadata.height,
        "resolution_matches": math.isclose(read_metadata.resolution_m, output_metadata.resolution_m, abs_tol=1e-9),
        "band_order_matches": read_metadata.band_names == output_metadata.band_names,
        "pixel_values_match": bool(np.allclose(read_array, sr.cpu().numpy(), atol=1e-6)),
    }
    for name, ok in checks.items():
        print(f"[verify] {name}: {'OK' if ok else '*** FAILED ***'}")

    # Stage 9 -- explicit invariant checks.
    footprint_drift = max(abs(a - b) for a, b in zip(read_metadata.bounds, input_geo["bounds"]))
    footprint_ok = footprint_drift <= FOOTPRINT_TOLERANCE_M
    print(f"[invariant] input footprint == output footprint: drift={footprint_drift:.3e} m -> "
          f"{'OK' if footprint_ok else '*** FAILED ***'}")

    input_pixel_size = NATIVE_RESOLUTION_M
    output_pixel_size = abs(read_metadata.transform[0])
    expected_output_pixel_size = input_pixel_size / RGBN_SCALE_FACTOR
    pixel_size_ok = math.isclose(output_pixel_size, expected_output_pixel_size, abs_tol=PIXEL_SIZE_TOLERANCE_M)
    print(f"[invariant] output pixel size ({output_pixel_size} m) == input pixel size / 4 "
          f"({expected_output_pixel_size} m) -> {'OK' if pixel_size_ok else '*** FAILED ***'}")

    save_preview(read_array, OUTPUTS_DIR / "sr_preview_rgb.png")

    all_ok = all(checks.values()) and footprint_ok and pixel_size_ok

    metadata_report = {
        "experiment": "baseline-geoexport (Phase 2)",
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
        "preprocessing_mask_coverage": preprocessed.mask.coverage(),
        "output_shape": list(sr.shape),
        "inference_seconds": round(inference_seconds, 4),
        "output_geospatial_metadata": {
            "crs": output_metadata.crs,
            "transform": list(output_metadata.transform),
            "bounds": list(output_metadata.bounds),
            "resolution_m": output_metadata.resolution_m,
            "width": output_metadata.width,
            "height": output_metadata.height,
            "band_names": list(output_metadata.band_names),
        },
        "reopened_geotiff_metadata": {
            "crs": read_metadata.crs,
            "transform": list(read_metadata.transform),
            "bounds": list(read_metadata.bounds),
            "resolution_m": read_metadata.resolution_m,
            "width": read_metadata.width,
            "height": read_metadata.height,
            "band_names": list(read_metadata.band_names),
            "nodata_value": read_metadata.nodata_value,
            "sr_variant": read_metadata.sr_variant,
            "acquisition_timestamp": read_metadata.acquisition_timestamp,
        },
        "verification_checks": checks,
        "invariants": {
            "footprint_drift_m": footprint_drift,
            "footprint_ok": footprint_ok,
            "output_pixel_size_m": output_pixel_size,
            "expected_output_pixel_size_m": expected_output_pixel_size,
            "pixel_size_ok": pixel_size_ok,
        },
        "all_checks_passed": all_ok,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "rasterio_version": __import__("rasterio").__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_report, f, indent=2)

    print(f"\n[done] GeoTIFF  -> {geotiff_path}")
    print(f"[done] preview  -> {OUTPUTS_DIR / 'sr_preview_rgb.png'}")
    print(f"[done] metadata -> {metadata_path}")
    print(f"[result] all checks passed: {all_ok}")

    if not all_ok:
        raise SystemExit(
            "One or more geospatial verification checks failed -- see "
            "metadata/run_metadata.json for details."
        )


if __name__ == "__main__":
    main()
