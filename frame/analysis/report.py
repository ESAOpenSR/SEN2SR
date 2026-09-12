"""Top-level entry point: `run_ndvi_analysis` -- Phase 6's downstream
vegetation-index demonstration.

Ties `frame.analysis.indices` (NDVI at both grids),
`frame.analysis.comparison` (native-vs-SR comparison on the common grid),
and `frame.analysis.uncertainty_overlay` (relating Phase 5's uncertainty to
that comparison) into one `NDVIAnalysisReport`.

SCIENTIFIC FRAMING -- read before using any number this produces
--------------------------------------------------------------------
The purpose of this analysis is NOT to prove that 2.5 m NDVI is ground
truth. It demonstrates that FRAME can provide a finer-grained vegetation-
index visualization while explicitly exposing model-stability uncertainty.
Every `NDVIAnalysisReport` carries `scientific_caveats`
(`SCIENTIFIC_CAVEATS` below) stating plainly:

  1. SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid,
     not a directly observed native 2.5 m vegetation measurement.
  2. The native 10 m NDVI and SR-derived NDVI are not independent ground
     truths -- the SR output was itself derived from (a transformation of)
     the same underlying 10 m observation.
  3. Agreement or disagreement between them measures internal consistency
     and downstream utility, not proof of physical accuracy.
  4. Uncertainty here is the Phase 5 relative model-stability proxy, not a
     calibrated probability of NDVI error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from frame.analysis.comparison import NDVIComparison, compare_ndvi_on_common_grid, downsample_ndvi_to_native_grid
from frame.analysis.indices import NDVIResult, compute_ndvi_from_stack
from frame.analysis.uncertainty_overlay import (
    UncertaintyWeightedNDVISummary,
    aggregate_uncertainty_overall,
    compute_uncertainty_weighted_ndvi_summary,
)
from frame.consistency.downsample import AREA_AVERAGE_POOL
from frame.preprocessing.masks import ValidityMask

SCIENTIFIC_CAVEATS: Tuple[str, str, str, str] = (
    "SR-derived 2.5 m NDVI is a learned inference on a 2.5 m pixel grid, "
    "not a directly observed native 2.5 m vegetation measurement.",
    "The native 10 m NDVI and SR-derived NDVI are not independent ground "
    "truths -- the SR output was itself derived from the same underlying "
    "10 m observation.",
    "Agreement or disagreement between them measures internal consistency "
    "and downstream utility, not proof of physical accuracy.",
    "Uncertainty here is the Phase 5 relative model-stability proxy, not a "
    "calibrated probability of NDVI error.",
)

NDVI_FORMULA = "NDVI = (NIR - Red) / (NIR + Red), where NIR = B08, Red = B04"


@dataclass(frozen=True)
class NDVIAnalysisReport:
    native_ndvi: NDVIResult
    sr_ndvi: NDVIResult
    ndvi_comparison: NDVIComparison
    common_grid_mask: np.ndarray
    uncertainty_overall_sr_grid: torch.Tensor
    uncertainty_overall_native_grid: torch.Tensor
    uncertainty_weighted_summary: UncertaintyWeightedNDVISummary
    scientific_caveats: Tuple[str, str, str, str]
    metadata: Dict[str, Any]


def run_ndvi_analysis(
    lr_reflectance: torch.Tensor,
    sr_mean_prediction: torch.Tensor,
    sr_std_prediction: torch.Tensor,
    *,
    band_names: Sequence[str],
    scale_factor: int,
    native_resolution_m: float = 10.0,
    lr_mask: Optional[np.ndarray] = None,
) -> NDVIAnalysisReport:
    """Compute NDVI at both grids, compare them, and relate the comparison
    to Phase 5's uncertainty output.

    Args:
        lr_reflectance: Native LR reflectance, shape (bands, H, W) -- the
            same tensor `frame.preprocessing.preprocess_rgbn` produces.
        sr_mean_prediction: Phase 5's `UncertaintyResult.mean_prediction`,
            shape (bands, H*scale_factor, W*scale_factor).
        sr_std_prediction: Phase 5's `UncertaintyResult.std_prediction`,
            same shape as ``sr_mean_prediction``.
        band_names: Band order for both stacks (identical for LR and SR in
            every FRAME experiment so far).
        scale_factor: The SR model's upscaling factor.
        native_resolution_m: The native grid's pixel size (default 10.0,
            the proven RGBN path's value).
        lr_mask: Optional (H, W) boolean validity mask for the native grid
            (e.g. from `frame.preprocessing.PreprocessedInput.mask`).
            Defaults to all-valid when not given.
    """
    if lr_mask is None:
        lr_mask = ValidityMask.all_valid(shape=tuple(lr_reflectance.shape[1:])).array
    else:
        lr_mask = np.asarray(lr_mask, dtype=bool)

    sr_resolution_m = native_resolution_m / scale_factor

    native_ndvi = compute_ndvi_from_stack(
        lr_reflectance,
        band_names=band_names,
        external_mask=lr_mask,
        resolution_m=native_resolution_m,
        grid_label="native_10m",
    )
    sr_ndvi = compute_ndvi_from_stack(
        sr_mean_prediction,
        band_names=band_names,
        resolution_m=sr_resolution_m,
        grid_label="sr_2_5m",
    )

    # A native-grid pixel is only valid for comparison if the native NDVI
    # is itself computable there AND every SR sub-pixel underneath it had a
    # computable NDVI too (conservative: any invalid sub-pixel invalidates
    # the whole downsampled block for comparison purposes).
    sr_valid_fraction_native = downsample_ndvi_to_native_grid(sr_ndvi.valid_mask.float(), scale_factor=scale_factor)
    combined_mask = native_ndvi.valid_mask.numpy() & (sr_valid_fraction_native.numpy() >= 0.999)

    ndvi_comparison = compare_ndvi_on_common_grid(
        native_ndvi.ndvi, sr_ndvi.ndvi, combined_mask, scale_factor=scale_factor
    )

    uncertainty_overall_sr_grid = aggregate_uncertainty_overall(sr_std_prediction)
    uncertainty_overall_native_grid = downsample_ndvi_to_native_grid(
        uncertainty_overall_sr_grid, scale_factor=scale_factor
    )

    uncertainty_weighted_summary = compute_uncertainty_weighted_ndvi_summary(
        uncertainty_overall_native_grid, ndvi_comparison.absolute_difference_map, combined_mask
    )

    metadata: Dict[str, Any] = {
        "ndvi_formula": NDVI_FORMULA,
        "band_names": list(band_names),
        "scale_factor": scale_factor,
        "native_resolution_m": native_resolution_m,
        "sr_resolution_m": sr_resolution_m,
        "resampling_method": AREA_AVERAGE_POOL,
        "native_valid_pixel_count": int(native_ndvi.valid_mask.sum().item()),
        "sr_valid_pixel_count": int(sr_ndvi.valid_mask.sum().item()),
        "comparison_valid_pixel_count": ndvi_comparison.comparison.valid_pixel_count,
    }

    return NDVIAnalysisReport(
        native_ndvi=native_ndvi,
        sr_ndvi=sr_ndvi,
        ndvi_comparison=ndvi_comparison,
        common_grid_mask=combined_mask,
        uncertainty_overall_sr_grid=uncertainty_overall_sr_grid,
        uncertainty_overall_native_grid=uncertainty_overall_native_grid,
        uncertainty_weighted_summary=uncertainty_weighted_summary,
        scientific_caveats=SCIENTIFIC_CAVEATS,
        metadata=metadata,
    )
