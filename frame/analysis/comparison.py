"""Comparing native-10m and SR-2.5m NDVI on a common grid (Phase 6).

Per this phase's explicit instruction: a 2.5 m NDVI map is never compared
directly against a 10 m NDVI map. The SR-grid NDVI is always reduced to the
native grid first, via `frame.consistency.downsample.downsample_to_lr_grid`
(Phase 3's area-average pooling -- reused here, not reimplemented, since it
is already this project's documented, "scientifically appropriate" method
for reducing an SR-grid raster back to its parent grid; see that module's
own docstring for why area-average pooling was chosen over e.g. bicubic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from frame.analysis.errors import ShapeMismatchError
from frame.consistency import ComputationStatus
from frame.consistency.downsample import AREA_AVERAGE_POOL, downsample_to_lr_grid


@dataclass(frozen=True)
class NDVIComparisonResult:
    status: ComputationStatus
    valid_pixel_count: int
    mean_abs_difference: Optional[float]
    rmse: Optional[float]
    max_abs_difference: Optional[float]
    resampling_method: str


@dataclass(frozen=True)
class NDVIComparison:
    comparison: NDVIComparisonResult
    sr_ndvi_downsampled: torch.Tensor  # (H, W), native grid
    absolute_difference_map: torch.Tensor  # (H, W), native grid


def downsample_ndvi_to_native_grid(sr_ndvi: torch.Tensor, *, scale_factor: int) -> torch.Tensor:
    """Reduce a (H, W) SR-grid NDVI map to the native grid via area-average pooling."""
    if sr_ndvi.ndim != 2:
        raise ShapeMismatchError(f"Expected a 2-D (H, W) NDVI map, got {sr_ndvi.ndim}-D shape {tuple(sr_ndvi.shape)}.")
    downsampled = downsample_to_lr_grid(sr_ndvi[None], scale_factor)  # add/remove a dummy band axis
    return downsampled[0]


def compare_ndvi_on_common_grid(
    native_ndvi: torch.Tensor,
    sr_ndvi: torch.Tensor,
    mask: np.ndarray,
    *,
    scale_factor: int,
) -> NDVIComparison:
    """Downsample ``sr_ndvi`` to the native grid and compare it against ``native_ndvi``.

    Args:
        native_ndvi: (H, W) NDVI computed directly from the native 10 m bands.
        sr_ndvi: (H*scale_factor, W*scale_factor) NDVI computed from the SR
            output -- reduced to the native grid before any comparison.
        mask: (H, W) boolean validity mask on the native grid (e.g. the
            logical AND of both NDVIResults' own valid_mask, resampled/
            aligned as the caller sees fit -- this function does not
            derive one itself).
        scale_factor: The SR model's upscaling factor.
    """
    if native_ndvi.ndim != 2:
        raise ShapeMismatchError(f"Expected native_ndvi to be 2-D (H, W), got shape {tuple(native_ndvi.shape)}.")
    if sr_ndvi.shape != (native_ndvi.shape[0] * scale_factor, native_ndvi.shape[1] * scale_factor):
        raise ShapeMismatchError(
            f"sr_ndvi shape {tuple(sr_ndvi.shape)} does not equal native_ndvi shape {tuple(native_ndvi.shape)} "
            f"times scale_factor={scale_factor}."
        )

    sr_downsampled = downsample_ndvi_to_native_grid(sr_ndvi, scale_factor=scale_factor)

    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape != tuple(native_ndvi.shape):
        raise ShapeMismatchError(f"Mask shape {mask_np.shape} does not match native_ndvi shape {tuple(native_ndvi.shape)}.")

    diff = (sr_downsampled - native_ndvi).detach().cpu().numpy().astype(np.float64)
    absolute_difference_map = torch.from_numpy(np.abs(diff)).float()

    valid_pixel_count = int(mask_np.sum())
    if valid_pixel_count == 0:
        comparison = NDVIComparisonResult(
            status=ComputationStatus.NOT_COMPUTABLE,
            valid_pixel_count=0,
            mean_abs_difference=None,
            rmse=None,
            max_abs_difference=None,
            resampling_method=AREA_AVERAGE_POOL,
        )
    else:
        valid_diff = diff[mask_np]
        comparison = NDVIComparisonResult(
            status=ComputationStatus.COMPUTABLE,
            valid_pixel_count=valid_pixel_count,
            mean_abs_difference=float(np.mean(np.abs(valid_diff))),
            rmse=float(np.sqrt(np.mean(valid_diff**2))),
            max_abs_difference=float(np.max(np.abs(valid_diff))),
            resampling_method=AREA_AVERAGE_POOL,
        )

    return NDVIComparison(
        comparison=comparison, sr_ndvi_downsampled=sr_downsampled, absolute_difference_map=absolute_difference_map
    )
