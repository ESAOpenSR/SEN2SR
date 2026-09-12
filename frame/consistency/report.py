"""Top-level entry point: `run_consistency_diagnostics`.

Combines the downsample-consistency check (`frame.consistency.band_discrepancy`)
and the band-wise spectral checks (`frame.consistency.spectral_ratios`) into
one `ConsistencyDiagnostics` record with enough metadata (`parameters`) to
reproduce the calculation later.

Cross-tile information (`frame.consistency.tiles`) is NOT computed here --
it is a separate, optional call the caller attaches via the `cross_tile`
argument when tile-level outputs are actually available (see
`frame.consistency.tiles`'s module docstring for why no real multi-tile
FRAME run exists yet to compute one from automatically).

This module never modifies the SR tensor it is given, and never converts its
output into a pass/fail verdict -- see docs/FRAME_TECHNICAL_SPEC.md Section
9.2/11 and this package's README for why: a numeric accept/reject threshold
requires an empirical discrepancy distribution this project does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from frame.consistency.band_discrepancy import (
    DownsampleConsistencyResult,
    downsample_consistency_from_arrays,
)
from frame.consistency.band_discrepancy import _validate_inputs as _validate_downsample_inputs
from frame.consistency.downsample import downsample_to_lr_grid
from frame.consistency.spectral_ratios import (
    SpectralIndexComparison,
    compute_b08_b04_ratio_comparison,
    compute_ndvi_comparison,
)
from frame.consistency.tiles import TileOverlapResult


@dataclass(frozen=True)
class ConsistencyDiagnostics:
    """One run's full spectral/self-consistency diagnostics report.

    ``cross_tile`` is ``None`` unless the caller explicitly supplies a
    precomputed `TileOverlapResult` (this function does not compute one
    itself -- see the module docstring).
    """

    downsample_consistency: DownsampleConsistencyResult
    ndvi_comparison: SpectralIndexComparison
    b08_b04_ratio_comparison: SpectralIndexComparison
    cross_tile: Optional[TileOverlapResult]
    parameters: Dict[str, Any]


def run_consistency_diagnostics(
    lr: torch.Tensor,
    sr: torch.Tensor,
    mask: np.ndarray,
    *,
    band_names: Sequence[str],
    scale_factor: int,
    cross_tile: Optional[TileOverlapResult] = None,
) -> ConsistencyDiagnostics:
    """Run every Phase 3 self-consistency diagnostic against one SR result.

    Args:
        lr: Original LR input tensor, shape (bands, H, W).
        sr: SR output tensor, shape (bands, H*scale_factor, W*scale_factor).
        mask: Boolean validity mask, shape (H, W) -- see
            `frame.preprocessing.ValidityMask.array`.
        band_names: Name of each band along axis 0, in order (must include
            "B04" and "B08" for the spectral-index comparisons).
        scale_factor: The SR model's upscaling factor.
        cross_tile: An optional, separately-computed
            `frame.consistency.tiles.TileOverlapResult`, attached as-is if
            given.

    Returns:
        A ConsistencyDiagnostics record.
    """
    _validate_downsample_inputs(lr, sr, mask, band_names, scale_factor)

    # Downsample ONCE and reuse the same reduced array for every diagnostic
    # below, so the downsample-consistency and spectral-ratio results are
    # guaranteed to be computed from literally the same reduced SR data.
    sr_down = downsample_to_lr_grid(sr, scale_factor)
    lr_np = lr.detach().cpu().numpy()
    sr_down_np = sr_down.detach().cpu().numpy()
    mask_np = np.asarray(mask, dtype=bool)

    downsample_result = downsample_consistency_from_arrays(
        lr_np, sr_down_np, mask_np, band_names=band_names, scale_factor=int(scale_factor)
    )
    ndvi_result = compute_ndvi_comparison(lr_np, sr_down_np, mask_np, band_names=band_names)
    ratio_result = compute_b08_b04_ratio_comparison(lr_np, sr_down_np, mask_np, band_names=band_names)

    parameters: Dict[str, Any] = {
        "downsample_method": downsample_result.downsample_method,
        "scale_factor": int(scale_factor),
        "band_names": tuple(band_names),
        "lr_shape": tuple(lr.shape),
        "sr_shape": tuple(sr.shape),
        "mask_shape": tuple(mask_np.shape),
        "mask_coverage": downsample_result.mask_coverage,
        "valid_pixel_count": downsample_result.valid_pixel_count,
        "total_pixel_count": downsample_result.total_pixel_count,
    }

    return ConsistencyDiagnostics(
        downsample_consistency=downsample_result,
        ndvi_comparison=ndvi_result,
        b08_b04_ratio_comparison=ratio_result,
        cross_tile=cross_tile,
        parameters=parameters,
    )
