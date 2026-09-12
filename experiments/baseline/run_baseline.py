"""
Baseline 0 — SEN2SRLite NonReference_RGBN_x4, reproducible smoke experiment.

This script does NOT modify, import internals of, or depend on any change to the
upstream `sen2sr/` package. It only drives the package the same way the project's
own README does: pull a Sentinel-2 L2A sample with `cubo`, load a pretrained model
manifest with `mlstac`, and run one forward pass with `torch`.

What this is for
-----------------
Turns the ad hoc Phase 0 smoke test into something we can re-run later and get the
SAME scene, SAME preprocessing, and a SAME-SHAPED result every time -- so future
work (Phase 1+) has a fixed point of comparison. It does not compute any accuracy
metric (there is no reference high-resolution image to compare against yet) --
it only proves the pipeline runs and records exactly what happened.

Stages
------
  1. Resolve compute device (GPU if available, else CPU).
  2. Download (or reuse a cached copy of) the SEN2SRLite NonReference_RGBN_x4
     model artifact from Hugging Face via `mlstac`.
  3. Fetch ONE deterministic Sentinel-2 L2A scene via `cubo`, using a single-day
     date window instead of a wide range -- this avoids pulling and scanning
     hundreds of timesteps just to land on a specific one.
  4. Preprocess exactly as the upstream README documents (reflectance scaling,
     NaN/Inf cleanup).
  5. Run inference (SR: 10 m -> 2.5 m, x4) and time it.
  6. Save the raw input/output tensors, four PNG visualizations, and a JSON
     metadata record describing the run.

Usage
-----
    <venv>/bin/python experiments/baseline/run_baseline.py

Re-running this script with no arguments reproduces the same scene and the same
preprocessing every time (the model's own inference is deterministic given fixed
weights and input; no random seeding is needed since nothing here is stochastic).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import cubo
import matplotlib.pyplot as plt
import mlstac
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Stage 0 -- fixed experiment configuration
#
# Every value here is the deterministic "Baseline 0" definition. Change these
# only when deliberately defining a new baseline (and rename the directory --
# don't silently mutate what "Baseline 0" means).
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
METADATA_DIR = BASE_DIR / "metadata"

# Area of interest -- same point used in the upstream README examples and in
# the Phase 0 smoke test, so results are directly comparable to that run.
AOI_LAT = 39.49152740347753
AOI_LON = -0.4308725142800361

# Single-day window: the Phase 0 baseline scan across the full 2023 date range
# found this acquisition (2023-01-15) to be cloud-free at this AOI. Narrowing
# the query to one day makes the fetch fast and the chosen scene deterministic,
# instead of downloading ~250 timesteps and picking one by a cloudiness heuristic.
SCENE_START_DATE = "2023-01-15"
SCENE_END_DATE = "2023-01-16"
SCENE_TIME_INDEX = 0  # deterministic: first (and, for this window, only-distinct) item

# NonReference_RGBN_x4 uses exactly these four 10 m bands, in this order.
BANDS = ["B04", "B03", "B02", "B08"]  # Red, Green, Blue, NIR
NATIVE_RESOLUTION_M = 10.0
EDGE_SIZE_PX = 128  # the model's native patch size -- no tiling required

MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"
MODEL_MANIFEST_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)

# Weights are cached OUTSIDE the repository (not committed, not vendored) so the
# experiment stays fast to re-run without bloating the git working tree with
# binary model artifacts. Override with the SEN2SR_BASELINE_WEIGHTS_DIR env var.
WEIGHTS_CACHE_DIR = Path(
    os.environ.get(
        "SEN2SR_BASELINE_WEIGHTS_DIR",
        Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN",
    )
)


# ---------------------------------------------------------------------------
# Stage 1 -- compute device
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage 2 -- model weights (download once, reuse on every later run)
# ---------------------------------------------------------------------------

def ensure_weights(cache_dir: Path) -> Path:
    """Download the model manifest/weights via mlstac if not already cached."""
    manifest_path = cache_dir / "mlm.json"
    if manifest_path.exists():
        print(f"[weights] using cached artifact at {cache_dir}")
        return cache_dir

    print(f"[weights] downloading {MODEL_MANIFEST_URL} -> {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    mlstac.download(file=MODEL_MANIFEST_URL, output_dir=str(cache_dir))
    return cache_dir


# ---------------------------------------------------------------------------
# Stage 3 -- deterministic Sentinel-2 L2A fetch
# ---------------------------------------------------------------------------

def fetch_scene():
    """Fetch a single, deterministic Sentinel-2 L2A cube (no wide-range scan)."""
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
            "the STAC catalog may have changed. Widen SCENE_START_DATE/"
            "SCENE_END_DATE to re-derive a deterministic timestep."
        )
    print(f"[scene] {n} catalog item(s) returned for this window; using index {SCENE_TIME_INDEX}")
    return da


# ---------------------------------------------------------------------------
# Stage 4 -- preprocessing (matches the upstream README exactly)
# ---------------------------------------------------------------------------

def preprocess(da, time_index: int, device: torch.device) -> torch.Tensor:
    """Reflectance-scale one timestep and clean up non-finite values."""
    raw = (da[time_index].compute().to_numpy() / 10_000).astype("float32")
    X = torch.from_numpy(raw).float().to(device)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X


# ---------------------------------------------------------------------------
# Stage 5 -- inference
# ---------------------------------------------------------------------------

def run_inference(model: torch.nn.Module, X: torch.Tensor):
    """Run one forward pass and time it (GPU-synchronized if applicable)."""
    if X.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(X.device)
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        superX = model(X[None]).squeeze(0)
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mem_mib = None
    if X.device.type == "cuda":
        peak_mem_mib = round(torch.cuda.max_memory_allocated(X.device) / (1024**2), 1)

    return superX, elapsed, peak_mem_mib


# ---------------------------------------------------------------------------
# Stage 6 -- outputs: raw tensors + visualizations + metadata
# ---------------------------------------------------------------------------

def to_rgb(img_chw: np.ndarray, stretch=(2, 98)) -> np.ndarray:
    """Percentile-stretch bands [0,1,2] (R,G,B = B04,B03,B02) for display only."""
    rgb = np.stack([img_chw[0], img_chw[1], img_chw[2]], axis=-1)
    lo, hi = np.percentile(rgb, stretch)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)


def save_visualizations(X: torch.Tensor, superX: torch.Tensor, outputs_dir: Path) -> None:
    X_np = X.cpu().numpy()
    superX_np = superX.cpu().numpy()

    # Bicubic-upsampled input, at the SAME pixel size as the SR output, purely
    # for a fair visual side-by-side -- this is NOT part of the model pipeline.
    lr_bicubic = torch.nn.functional.interpolate(
        X.cpu()[None], size=superX_np.shape[-2:], mode="bicubic", antialias=True
    )[0].numpy()

    rgb_lr = to_rgb(X_np)
    rgb_lr_bicubic = to_rgb(lr_bicubic)
    rgb_sr = to_rgb(superX_np)

    # (a) LR input, native size
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_lr)
    plt.axis("off")
    plt.title(f"LR input (native)\n{X_np.shape[-1]}x{X_np.shape[-2]} px @ {NATIVE_RESOLUTION_M:.0f} m")
    plt.tight_layout()
    plt.savefig(outputs_dir / "lr_input_rgb.png", dpi=150)
    plt.close()

    # (b) LR input, bicubic-upsampled to output size (display baseline only)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_lr_bicubic)
    plt.axis("off")
    plt.title("LR input, bicubic-upsampled\n(display baseline, not a model output)")
    plt.tight_layout()
    plt.savefig(outputs_dir / "lr_bicubic_rgb.png", dpi=150)
    plt.close()

    # (c) SR result
    out_res_m = NATIVE_RESOLUTION_M * X_np.shape[-1] / superX_np.shape[-1]
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_sr)
    plt.axis("off")
    plt.title(f"SR output ({MODEL_NAME})\n{superX_np.shape[-1]}x{superX_np.shape[-2]} px @ {out_res_m:.2f} m")
    plt.tight_layout()
    plt.savefig(outputs_dir / "sr_result_rgb.png", dpi=150)
    plt.close()

    # (d) side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    axes[0].imshow(rgb_lr_bicubic)
    axes[0].set_title("Bicubic-upsampled input (display baseline)")
    axes[0].axis("off")
    axes[1].imshow(rgb_sr)
    axes[1].set_title(f"{MODEL_NAME} output")
    axes[1].axis("off")
    fig.suptitle(
        f"Baseline 0 -- Sentinel-2 L2A near ({AOI_LAT:.4f}, {AOI_LON:.4f}), "
        f"{SCENE_START_DATE} -- {NATIVE_RESOLUTION_M:.0f} m -> {out_res_m:.2f} m"
    )
    plt.tight_layout()
    plt.savefig(outputs_dir / "comparison_side_by_side.png", dpi=150)
    plt.close()


def build_metadata(
    *,
    da,
    X: torch.Tensor,
    superX: torch.Tensor,
    device: torch.device,
    inference_seconds: float,
    peak_mem_mib,
    weight_dir: Path,
) -> dict:
    scale_factor = superX.shape[-1] / X.shape[-1]
    return {
        "experiment": "baseline-0",
        "model_name": MODEL_NAME,
        "model_artifact_source": MODEL_MANIFEST_URL,
        "model_weights_cache_dir": str(weight_dir),
        "coordinates": {"lat": AOI_LAT, "lon": AOI_LON},
        "scene_date_window": {"start": SCENE_START_DATE, "end": SCENE_END_DATE},
        "scene_time_index": SCENE_TIME_INDEX,
        "scene_timestamp": str(da["time"].values[SCENE_TIME_INDEX]),
        "bands": BANDS,
        "input_resolution_m": NATIVE_RESOLUTION_M,
        "output_resolution_m": round(NATIVE_RESOLUTION_M / scale_factor, 4),
        "scale_factor": scale_factor,
        "input_shape": list(X.shape),
        "output_shape": list(superX.shape),
        "inference_seconds": round(inference_seconds, 4),
        "peak_gpu_memory_mib": peak_mem_mib,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


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
    X = preprocess(da, SCENE_TIME_INDEX, device)
    print(f"[preprocess] input tensor shape: {tuple(X.shape)}")

    model = mlstac.load(str(weight_dir)).compiled_model(device=device)
    superX, inference_seconds, peak_mem_mib = run_inference(model, X)
    print(f"[inference] {inference_seconds:.3f}s, output shape {tuple(superX.shape)}")

    # raw tensors
    torch.save(X.cpu(), OUTPUTS_DIR / "input_tensor.pt")
    torch.save(superX.cpu(), OUTPUTS_DIR / "sr_tensor.pt")

    # visualizations
    save_visualizations(X, superX, OUTPUTS_DIR)

    # metadata
    metadata = build_metadata(
        da=da,
        X=X,
        superX=superX,
        device=device,
        inference_seconds=inference_seconds,
        peak_mem_mib=peak_mem_mib,
        weight_dir=weight_dir,
    )
    metadata_path = METADATA_DIR / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[done] outputs  -> {OUTPUTS_DIR}")
    print(f"[done] metadata -> {metadata_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
