"""Phase 1 experiment -- FRAME preprocessing layer vs. Baseline 0.

Goal
----
Prove that `frame.preprocessing.preprocess_rgbn` is a faithful, more robust
*replacement* for the ad hoc inline preprocessing in
`experiments/baseline/run_baseline.py` -- not an improvement in SR quality,
just a numerically equivalent, better-validated substitute.

This script does NOT modify `experiments/baseline/` in any way. It reuses
Baseline 0's exact deterministic scene definition (same AOI, same date
window, same time index, same bands, same model) as a fixed comparison
point, and reads Baseline 0's already-saved reference tensors from
`experiments/baseline/outputs/` rather than recomputing them.

What this proves
-----------------
1. Our preprocessing layer, run on the SAME raw fetched scene, produces a
   model-input tensor numerically equivalent (within a documented
   floating-point tolerance) to Baseline 0's proven input tensor -- whether
   the raw bands arrive already in canonical order, or scrambled (exercising
   the reorder-bands logic on real data, not just synthetic arrays).
2. Feeding our preprocessed tensor into the SAME unmodified upstream model
   used by Baseline 0 produces an SR output numerically equivalent to
   Baseline 0's saved SR output.

What this does NOT prove
-------------------------
Nothing about SR accuracy or quality -- there is still no reference
high-resolution image here (see docs/FRAME_TECHNICAL_SPEC.md Section 11).
This experiment is about preprocessing-layer correctness only.

Stages
------
  1. Resolve compute device (mirrors Baseline 0).
  2. Reuse the SAME cached SEN2SRLite/NonReference_RGBN_x4 model artifact
     Baseline 0 already downloaded (no re-download).
  3. Fetch the SAME deterministic Sentinel-2 L2A scene Baseline 0 used.
  4. Derive geospatial metadata (CRS, affine transform, bounds) from the
     fetched cube -- cubo deliberately strips stackstac's own crs/transform
     attrs (see cubo.cubo.create), so this is recomputed from the cube's
     epsg attribute and its x/y pixel-center coordinate arrays.
  5. Run the fetched raw array through frame.preprocessing.preprocess_rgbn
     twice: once with bands already in canonical order, once with bands
     deliberately scrambled beforehand (both derived from the identical
     fetched array, so any difference is attributable only to our code).
  6. Load Baseline 0's saved input_tensor.pt / sr_tensor.pt and compare.
  7. Run the unmodified upstream model on our preprocessed tensor and
     compare its output against Baseline 0's saved SR output.
  8. Save a numeric comparison report, a diff visualization, and metadata.

Usage
-----
    <venv>/bin/python experiments/baseline_preprocessing/run_experiment.py
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

# `frame` is a local, non-installed package living at the repository root
# (two levels above this file: experiments/baseline_preprocessing/<this
# file> -> experiments/ -> repo root). Make it importable without requiring
# a package install step or a manually-set PYTHONPATH, so this script stays
# runnable the same simple way Baseline 0's own script is documented to run.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cubo
import matplotlib.pyplot as plt
import mlstac
import torch

from frame.preprocessing import PreprocessedInput, RGBN_BANDS, preprocess_rgbn

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
#
# These values MUST match experiments/baseline/run_baseline.py exactly --
# that is what makes this "the same deterministic scene." Do not change
# these independently of that file.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METADATA_DIR = BASE_DIR / "metadata"

BASELINE0_DIR = BASE_DIR.parent / "baseline"
BASELINE0_INPUT_TENSOR = BASELINE0_DIR / "outputs" / "input_tensor.pt"
BASELINE0_SR_TENSOR = BASELINE0_DIR / "outputs" / "sr_tensor.pt"

AOI_LAT = 39.49152740347753
AOI_LON = -0.4308725142800361

SCENE_START_DATE = "2023-01-15"
SCENE_END_DATE = "2023-01-16"
SCENE_TIME_INDEX = 0

BANDS = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR -- already RGBN_BANDS order
assert list(RGBN_BANDS) == BANDS, "frame.preprocessing.RGBN_BANDS drifted from Baseline 0's BANDS"

NATIVE_RESOLUTION_M = 10.0
EDGE_SIZE_PX = 128

MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"
MODEL_MANIFEST_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)

# Reuse the exact same cached artifact Baseline 0 downloaded -- same model,
# no reason to fetch or store it twice.
WEIGHTS_CACHE_DIR = Path(
    os.environ.get(
        "SEN2SR_BASELINE_WEIGHTS_DIR",
        Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN",
    )
)

# Documented floating-point comparison tolerances (NOT a scientific-accuracy
# threshold -- see docs/FRAME_TECHNICAL_SPEC.md's instruction not to invent
# those; this only bounds acceptable floating-point/library-implementation
# noise between two code paths computing the same arithmetic).
INPUT_TENSOR_ATOL = 1e-5
SR_OUTPUT_ATOL = 1e-4


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
# Stage 4 -- geospatial metadata extraction
#
# cubo.create() deliberately deletes stackstac's own "crs"/"transform"
# attrs (see cubo/cubo.py: `attributes = ["spec", "crs", "transform",
# "resolution"]; for attribute in attributes: del cube.attrs[attribute]`)
# and only keeps `epsg` and `resolution`. The affine transform and bounds
# are recomputed here from those plus the cube's own pixel-center x/y
# coordinate arrays -- standard north-up raster geometry, not a guess.
# ---------------------------------------------------------------------------

def extract_geospatial_metadata(da, resolution_m: float):
    epsg = da.attrs["epsg"]
    x = da["x"].values
    y = da["y"].values

    # x/y are pixel-CENTER coordinates; the affine transform's origin is the
    # upper-left CORNER of the upper-left pixel, half a pixel outside the
    # first center coordinate on each axis. y descends (north-up).
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

    return {
        "crs": f"EPSG:{epsg}",
        "transform": transform,
        "bounds": bounds,
    }


# ---------------------------------------------------------------------------
# Stage 5/6/7 -- comparison helpers
# ---------------------------------------------------------------------------

def compare_tensors(name: str, ours: torch.Tensor, reference: torch.Tensor, atol: float) -> dict:
    ours_cpu = ours.detach().cpu().to(torch.float64)
    reference_cpu = reference.detach().cpu().to(torch.float64)
    diff = (ours_cpu - reference_cpu).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()
    is_close = torch.allclose(ours_cpu, reference_cpu, atol=atol, rtol=0.0)
    result = {
        "name": name,
        "shapes_match": list(ours.shape) == list(reference.shape),
        "atol": atol,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "numerically_equivalent": bool(is_close),
    }
    status = "EQUIVALENT" if is_close else "*** DIVERGED ***"
    print(f"[compare] {name}: max_abs_diff={max_abs_diff:.3e} mean_abs_diff={mean_abs_diff:.3e} atol={atol:.1e} -> {status}")
    return result


def save_diff_visualization(ours_sr: torch.Tensor, baseline_sr: torch.Tensor, path: Path) -> None:
    diff = (ours_sr.cpu().float() - baseline_sr.cpu().float()).abs().numpy()
    diff_map = diff.sum(axis=0)  # sum over bands -> (H, W)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5))
    im = ax.imshow(diff_map, cmap="magma")
    ax.set_title("abs(ours SR - Baseline 0 SR), summed over bands", fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    if not BASELINE0_INPUT_TENSOR.exists() or not BASELINE0_SR_TENSOR.exists():
        raise RuntimeError(
            f"Baseline 0 outputs not found at {BASELINE0_DIR / 'outputs'}. "
            "Run experiments/baseline/run_baseline.py first -- this experiment "
            "compares against its saved tensors rather than modifying it."
        )

    device = get_device()
    print(f"[device] {device}")

    weight_dir = ensure_weights(WEIGHTS_CACHE_DIR)

    da = fetch_scene()
    raw = da[SCENE_TIME_INDEX].compute().to_numpy()  # (bands, H, W), raw digital numbers, band order == BANDS
    geo = extract_geospatial_metadata(da, NATIVE_RESOLUTION_M)
    timestamp = str(da["time"].values[SCENE_TIME_INDEX])
    print(f"[scene] geospatial metadata: {geo}")

    common_kwargs = dict(
        input_scale="raw_digital_number",
        resolution_m=NATIVE_RESOLUTION_M,
        nodata_value=0.0,  # Sentinel-2 L2A's standard nodata convention
        crs=geo["crs"],
        transform=geo["transform"],
        bounds=geo["bounds"],
        acquisition_timestamp=timestamp,
        require_geospatial=True,
    )

    # (a) bands already in canonical order -- the same order Baseline 0 fetched
    result_in_order: PreprocessedInput = preprocess_rgbn(raw, band_names=BANDS, **common_kwargs)

    # (b) the SAME raw array, but deliberately scrambled before preprocessing,
    # to exercise reorder_bands on real (not synthetic) data
    scramble_order = ["B08", "B02", "B04", "B03"]
    scramble_indices = [BANDS.index(b) for b in scramble_order]
    scrambled_raw = raw[scramble_indices, ...]
    result_scrambled: PreprocessedInput = preprocess_rgbn(
        scrambled_raw, band_names=scramble_order, **common_kwargs
    )

    print(f"[preprocess] in-order mask coverage:  {result_in_order.mask.coverage():.6f}")
    print(f"[preprocess] scrambled mask coverage: {result_scrambled.mask.coverage():.6f}")

    baseline_input = torch.load(BASELINE0_INPUT_TENSOR, weights_only=True)
    baseline_sr = torch.load(BASELINE0_SR_TENSOR, weights_only=True)

    comparisons = [
        compare_tensors("in_order_input_vs_baseline0_input", result_in_order.tensor, baseline_input, INPUT_TENSOR_ATOL),
        compare_tensors("scrambled_input_vs_baseline0_input", result_scrambled.tensor, baseline_input, INPUT_TENSOR_ATOL),
        compare_tensors("in_order_input_vs_scrambled_input", result_in_order.tensor, result_scrambled.tensor, INPUT_TENSOR_ATOL),
    ]

    # Run the SAME unmodified upstream model on our preprocessed tensor.
    model = mlstac.load(str(weight_dir)).compiled_model(device=device)
    X = result_in_order.tensor.to(device)
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        our_sr = model(X[None]).squeeze(0)
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    inference_seconds = time.time() - t0
    print(f"[inference] {inference_seconds:.3f}s, output shape {tuple(our_sr.shape)}")

    comparisons.append(
        compare_tensors("our_sr_output_vs_baseline0_sr_output", our_sr, baseline_sr.to(device), SR_OUTPUT_ATOL)
    )

    torch.save(result_in_order.tensor.cpu(), OUTPUTS_DIR / "input_tensor.pt")
    torch.save(our_sr.cpu(), OUTPUTS_DIR / "sr_tensor.pt")
    save_diff_visualization(our_sr, baseline_sr.to(device), OUTPUTS_DIR / "sr_diff_vs_baseline0.png")

    all_equivalent = all(c["numerically_equivalent"] for c in comparisons)

    metadata = {
        "experiment": "baseline-preprocessing (Phase 1)",
        "compares_against": str(BASELINE0_DIR),
        "model_name": MODEL_NAME,
        "model_artifact_source": MODEL_MANIFEST_URL,
        "model_weights_cache_dir": str(weight_dir),
        "coordinates": {"lat": AOI_LAT, "lon": AOI_LON},
        "scene_date_window": {"start": SCENE_START_DATE, "end": SCENE_END_DATE},
        "scene_time_index": SCENE_TIME_INDEX,
        "scene_timestamp": timestamp,
        "bands": BANDS,
        "input_resolution_m": NATIVE_RESOLUTION_M,
        "geospatial_metadata": geo,
        "preprocessing_mask_coverage": {
            "in_order": result_in_order.mask.coverage(),
            "scrambled": result_scrambled.mask.coverage(),
        },
        "output_shape": list(our_sr.shape),
        "inference_seconds": round(inference_seconds, 4),
        "comparisons": comparisons,
        "all_numerically_equivalent": all_equivalent,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[done] outputs  -> {OUTPUTS_DIR}")
    print(f"[done] metadata -> {metadata_path}")
    print(f"[result] all comparisons numerically equivalent: {all_equivalent}")

    if not all_equivalent:
        raise SystemExit(
            "One or more comparisons against Baseline 0 diverged beyond the "
            "documented tolerance -- see metadata/run_metadata.json for details."
        )


if __name__ == "__main__":
    main()
