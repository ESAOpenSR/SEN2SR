"""Relating Phase 5's uncertainty output to NDVI disagreement (Phase 6).

`aggregate_uncertainty_overall` collapses the 4-band `std_prediction`
(`frame.uncertainty.UncertaintyResult.std_prediction`) to a single map by
averaging over bands -- the exact same convention Phase 5's own experiment
script already uses for its overall uncertainty visualization
(`std_prediction.mean(dim=0)`), not a new one invented here.

`compute_uncertainty_weighted_ndvi_summary` asks, without ever picking a
"high uncertainty" cutoff: does the model-stability uncertainty signal
track where the native/SR NDVI comparison disagrees most? Two threshold-free
ways to ask that:

  - `correlation_uncertainty_vs_abs_diff` -- the plain Pearson correlation
    between the (per-pixel) uncertainty map and the (per-pixel) absolute
    NDVI difference map, on the common grid. A positive value says
    "yes, roughly" without needing to define what counts as "high."
  - `uncertainty_weighted_mean_abs_diff` -- the mean absolute NDVI
    difference, weighted by the uncertainty map itself (higher-uncertainty
    pixels contribute more), compared directly against the plain
    (unweighted) mean for the same pixels.

Neither of these is a confidence threshold, an accept/reject rule, or a
calibrated probability -- see frame/uncertainty/README.md and
frame/analysis/README.md for why none is invented anywhere in this project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from frame.analysis.errors import ShapeMismatchError
from frame.consistency import ComputationStatus


def aggregate_uncertainty_overall(std_prediction: torch.Tensor) -> torch.Tensor:
    """Mean over the band axis -- (bands, H, W) -> (H, W).

    Matches `experiments/uncertainty/run_experiment.py`'s own
    `overall_std_map = result.std_prediction.mean(dim=0)` exactly, so the
    overlay/summary computed here is directly comparable to Phase 5's own
    reported numbers on the same scene.
    """
    if std_prediction.ndim != 3:
        raise ShapeMismatchError(
            f"Expected a 3-D (bands, H, W) tensor, got {std_prediction.ndim}-D shape {tuple(std_prediction.shape)}."
        )
    return std_prediction.mean(dim=0)


@dataclass(frozen=True)
class UncertaintyWeightedNDVISummary:
    status: ComputationStatus
    valid_pixel_count: int
    correlation_uncertainty_vs_abs_diff: Optional[float]
    uncertainty_weighted_mean_abs_diff: Optional[float]
    unweighted_mean_abs_diff: Optional[float]


def compute_uncertainty_weighted_ndvi_summary(
    uncertainty_map: torch.Tensor,
    absolute_difference_map: torch.Tensor,
    mask: np.ndarray,
) -> UncertaintyWeightedNDVISummary:
    """Relate a per-pixel uncertainty map to a per-pixel NDVI absolute-
    difference map, both on the SAME grid.

    Args:
        uncertainty_map: (H, W) overall model-stability uncertainty, on the
            same grid as ``absolute_difference_map`` (the caller is
            responsible for having reduced both to a common grid -- see
            `frame.analysis.comparison.downsample_ndvi_to_native_grid`,
            reused for the uncertainty map too).
        absolute_difference_map: (H, W) |downsampled SR NDVI - native NDVI|.
        mask: (H, W) boolean validity mask.
    """
    if uncertainty_map.shape != absolute_difference_map.shape:
        raise ShapeMismatchError(
            f"uncertainty_map shape {tuple(uncertainty_map.shape)} != "
            f"absolute_difference_map shape {tuple(absolute_difference_map.shape)}."
        )
    mask_np = np.asarray(mask, dtype=bool)
    if mask_np.shape != tuple(uncertainty_map.shape):
        raise ShapeMismatchError(f"Mask shape {mask_np.shape} does not match map shape {tuple(uncertainty_map.shape)}.")

    valid_pixel_count = int(mask_np.sum())
    if valid_pixel_count == 0:
        return UncertaintyWeightedNDVISummary(
            status=ComputationStatus.NOT_COMPUTABLE,
            valid_pixel_count=0,
            correlation_uncertainty_vs_abs_diff=None,
            uncertainty_weighted_mean_abs_diff=None,
            unweighted_mean_abs_diff=None,
        )

    u = uncertainty_map.detach().cpu().numpy().astype(np.float64)[mask_np]
    d = absolute_difference_map.detach().cpu().numpy().astype(np.float64)[mask_np]

    unweighted_mean_abs_diff = float(np.mean(d))

    if np.std(u) > 0 and np.std(d) > 0:
        correlation = float(np.corrcoef(u, d)[0, 1])
    else:
        correlation = None  # zero variance in either series -- correlation is mathematically undefined, not zero

    uncertainty_sum = float(np.sum(u))
    uncertainty_weighted_mean_abs_diff = float(np.sum(u * d) / uncertainty_sum) if uncertainty_sum > 0 else None

    return UncertaintyWeightedNDVISummary(
        status=ComputationStatus.COMPUTABLE,
        valid_pixel_count=valid_pixel_count,
        correlation_uncertainty_vs_abs_diff=correlation,
        uncertainty_weighted_mean_abs_diff=uncertainty_weighted_mean_abs_diff,
        unweighted_mean_abs_diff=unweighted_mean_abs_diff,
    )
