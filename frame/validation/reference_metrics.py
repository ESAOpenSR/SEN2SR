"""Standard reference-based image-quality metrics (docs/FRAME_TECHNICAL_SPEC.md
Section 12): PSNR, SSIM, RMSE, SAM, ERGAS.

This is group "(A) standard reference-based metrics" in the Phase 4 report
(see frame/validation/report.py) -- computed between an *estimate* (this
module doesn't care whether it's the bicubic baseline or the real SEN2SR
output -- the caller decides which, and calls this function once for each)
and a genuine higher-resolution reference image. Distinct from opensr-test's
own metric vocabulary (group B, frame/validation/opensr_test_metrics.py) and
from frame.consistency's no-reference self-consistency diagnostics (group C).

Where each formula comes from, and why
---------------------------------------
- **RMSE, PSNR**: the underlying mean-squared-error is computed via
  `opensr_test.distance.L2` (installed as a real dependency, not copied --
  see frame/validation/README.md) in its per-pixel mode, so the raw error
  computation is the benchmark's own tested implementation. PSNR itself is
  then the field-standard `10*log10(data_range**2 / mse)` -- deliberately
  NOT `opensr_test`'s own `"psnr"` distance method, which computes the
  *reciprocal* (`opensr_test.distance.IPSNR`, "Inverse PSNR", designed to
  fit their all-metrics-are-a-lower-is-better-distance convention). Reusing
  that inverted value under the name "PSNR" here would silently redefine a
  metric every reader of this report expects in its standard,
  higher-is-better, decibel form.
- **SAM**: `opensr_test.distance.SAD` ("Spectral Angle Distance") in its
  per-pixel mode -- its own source computes exactly the textbook Spectral
  Angle Mapper formula (`arccos(dot(x,y) / (||x|| * ||y||))`, in degrees;
  the library's own internal variable is even named `sam_score`). Reused
  directly, not reimplemented -- the only difference from calling their
  `Metrics` class is that masking is applied here afterward (their classes
  have no mask parameter), and that this module always compares against the
  real HR reference, not the LR self-consistency pairing their `Metrics
  .consistency()` uses by default.
- **SSIM**: `skimage.metrics.structural_similarity` (a dependency already
  pulled in transitively by `opensr-test` itself) -- not reimplemented,
  since `opensr-test` does not provide an SSIM metric at all.
- **ERGAS**: not provided by either `opensr-test` or `scikit-image`.
  Implemented directly here from its standard, published form (Wald, 2000;
  the field-standard pansharpening/fusion-quality formula
  docs/FRAME_TECHNICAL_SPEC.md Section 12 already names):
  `ERGAS = 100 * (1/scale_factor) * sqrt(mean_over_bands((RMSE_band / mean(HR_band))**2))`.

Masking
-------
Every metric here is computed only over pixels where the caller-supplied
boolean mask is True -- masked (cloud/nodata/border) pixels are excluded
from every sum/mean, never treated as valid ground truth. See
frame/validation/README.md for exactly what mask each experiment supplies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from opensr_test.distance import L2, SAD
from skimage.metrics import structural_similarity

from frame.consistency.status import ComputationStatus
from frame.validation.errors import ShapeMismatchError


@dataclass(frozen=True)
class ReferenceMetrics:
    status: ComputationStatus
    valid_pixel_count: int
    band_names: Tuple[str, ...]
    scale_factor: int
    data_range: float
    psnr_db: Optional[float]
    ssim: Optional[float]
    rmse: Optional[float]
    sam_degrees: Optional[float]
    ergas: Optional[float]


def _validate(estimate: torch.Tensor, hr: torch.Tensor, mask: np.ndarray, band_names: Sequence[str]) -> None:
    if estimate.ndim != 3 or hr.ndim != 3:
        raise ShapeMismatchError(
            f"Expected 3-D (bands, H, W) tensors, got estimate={tuple(estimate.shape)}, hr={tuple(hr.shape)}."
        )
    if estimate.shape != hr.shape:
        raise ShapeMismatchError(f"estimate shape {tuple(estimate.shape)} != hr shape {tuple(hr.shape)}.")
    if len(band_names) != estimate.shape[0]:
        raise ShapeMismatchError(
            f"band_names has {len(band_names)} entries but the tensors have {estimate.shape[0]} band(s)."
        )
    mask_arr = np.asarray(mask)
    if mask_arr.shape != tuple(estimate.shape[1:]):
        raise ShapeMismatchError(f"Mask shape {mask_arr.shape} does not match the image shape {tuple(estimate.shape[1:])}.")


def _ergas(
    estimate_np: np.ndarray, hr_np: np.ndarray, mask_np: np.ndarray, *, scale_factor: int
) -> float:
    """100 * (1/scale_factor) * sqrt(mean_over_bands((RMSE_band / mean(HR_band))**2))."""
    n_bands = estimate_np.shape[0]
    ratios = []
    for b in range(n_bands):
        est_b = estimate_np[b][mask_np].astype(np.float64)
        hr_b = hr_np[b][mask_np].astype(np.float64)
        rmse_b = float(np.sqrt(np.mean((est_b - hr_b) ** 2)))
        mean_b = float(np.mean(hr_b))
        ratios.append((rmse_b / mean_b) ** 2 if mean_b != 0 else 0.0)
    return 100.0 * (1.0 / scale_factor) * math.sqrt(float(np.mean(ratios)))


def compute_reference_metrics(
    estimate: torch.Tensor,
    hr: torch.Tensor,
    mask: np.ndarray,
    *,
    band_names: Sequence[str],
    scale_factor: int,
    data_range: float = 1.0,
) -> ReferenceMetrics:
    """Compute PSNR, SSIM, RMSE, SAM, and ERGAS between ``estimate`` and ``hr``.

    Args:
        estimate: The image being evaluated -- bicubic baseline OR SR
            output; this function is identical either way. Shape (C, H, W).
        hr: The real, independently-sourced high-resolution reference.
            Shape (C, H, W), same as ``estimate``.
        mask: Boolean validity mask, shape (H, W). False pixels (cloud/
            nodata/border) are excluded from every metric.
        band_names: Name of each band along axis 0, in order.
        scale_factor: The SR scale factor this comparison is at (needed by
            ERGAS's resolution-ratio term).
        data_range: The assumed value range for PSNR/SSIM (1.0 for
            reflectance-scaled data, the convention used throughout FRAME).
    """
    _validate(estimate, hr, mask, band_names)
    mask_np = np.asarray(mask, dtype=bool)
    valid_pixel_count = int(mask_np.sum())

    if valid_pixel_count == 0:
        return ReferenceMetrics(
            status=ComputationStatus.NOT_COMPUTABLE,
            valid_pixel_count=0,
            band_names=tuple(band_names),
            scale_factor=int(scale_factor),
            data_range=float(data_range),
            psnr_db=None,
            ssim=None,
            rmse=None,
            sam_degrees=None,
            ergas=None,
        )

    estimate_f = estimate.float()
    hr_f = hr.float()

    # RMSE / PSNR -- reuse opensr_test's own per-pixel MSE computation.
    mse_map = L2(x=estimate_f, y=hr_f, method="pixel").compute().detach().cpu().numpy()
    mse = float(mse_map[mask_np].mean())
    rmse = math.sqrt(mse)
    psnr_db = 10.0 * math.log10((data_range**2) / mse) if mse > 0 else float("inf")

    # SAM -- reuse opensr_test's own per-pixel spectral-angle computation.
    sam_map = SAD(x=estimate_f, y=hr_f, method="pixel").compute().detach().cpu().numpy()
    sam_degrees = float(sam_map[mask_np].mean())

    # SSIM -- via scikit-image (opensr-test provides no SSIM).
    estimate_np = estimate_f.detach().cpu().numpy()
    hr_np = hr_f.detach().cpu().numpy()
    _, ssim_full = structural_similarity(
        hr_np, estimate_np, channel_axis=0, data_range=data_range, full=True
    )
    ssim_map = ssim_full.mean(axis=0)  # average over bands -> (H, W)
    ssim = float(ssim_map[mask_np].mean())

    # ERGAS -- standard formula, implemented directly (no existing tool provides it).
    ergas = _ergas(estimate_np, hr_np, mask_np, scale_factor=scale_factor)

    return ReferenceMetrics(
        status=ComputationStatus.COMPUTABLE,
        valid_pixel_count=valid_pixel_count,
        band_names=tuple(band_names),
        scale_factor=int(scale_factor),
        data_range=float(data_range),
        psnr_db=psnr_db,
        ssim=ssim,
        rmse=rmse,
        sam_degrees=sam_degrees,
        ergas=ergas,
    )
