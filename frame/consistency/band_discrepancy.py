"""Downsample-consistency diagnostic (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).

Compares an SR output, reduced back to its LR grid via
`frame.consistency.downsample.downsample_to_lr_grid`, against the real LR
observation it was derived from.

This is a SELF-CONSISTENCY test, not a ground-truth accuracy metric: there is
no independent higher-resolution reference here (see
docs/FRAME_TECHNICAL_SPEC.md Section 11). A large discrepancy is a signal
that *something* in the pipeline deserves attention -- the model's own
Fourier hard constraint (`sen2sr/models/tricks.py`, unmodified, still runs
internally on every inference), a masking bug, a band-order mismatch, or a
numerically degenerate tile -- not proof the SR output is "wrong" in any
absolute sense. No numeric pass/fail threshold is applied here; see
`frame.consistency.status.ComputationStatus` and this package's README.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from frame.consistency.downsample import AREA_AVERAGE_POOL, downsample_to_lr_grid
from frame.consistency.errors import InvalidMaskError, ShapeMismatchError
from frame.consistency.status import ComputationStatus


@dataclass(frozen=True)
class BandDiscrepancy:
    """Discrepancy statistics for one band (or the pooled 'overall' result).

    ``valid_pixel_count`` is always the number of valid *spatial* locations
    (identical across every band and the overall result, since the validity
    mask is shared spatially). ``sample_count`` is the actual number of
    scalar error values the statistics below were computed from -- equal to
    ``valid_pixel_count`` for a single band, and to
    ``valid_pixel_count * number_of_bands`` for the pooled 'overall' result
    -- recorded explicitly so the calculation is reproducible without having
    to re-derive it.
    """

    band_name: Optional[str]
    status: ComputationStatus
    valid_pixel_count: int
    sample_count: int
    mean_abs_error: Optional[float]
    rmse: Optional[float]
    max_abs_error: Optional[float]
    normalized_rmse: Optional[float]


@dataclass(frozen=True)
class DownsampleConsistencyResult:
    downsample_method: str
    scale_factor: int
    band_names: Tuple[str, ...]
    valid_pixel_count: int
    total_pixel_count: int
    mask_coverage: float
    overall: BandDiscrepancy
    per_band: Dict[str, BandDiscrepancy]


def _stats(errors: np.ndarray, lr_values: np.ndarray) -> Tuple[float, float, float, Optional[float]]:
    abs_errors = np.abs(errors)
    mean_abs = float(abs_errors.mean())
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_abs = float(abs_errors.max())
    lr_mean_abs = float(np.abs(lr_values).mean())
    normalized_rmse = float(rmse / lr_mean_abs) if lr_mean_abs > 0 else None
    return mean_abs, rmse, max_abs, normalized_rmse


def _band_discrepancy(
    band_name: Optional[str], lr_values: np.ndarray, sr_down_values: np.ndarray, valid_pixel_count: int
) -> BandDiscrepancy:
    """``lr_values``/``sr_down_values`` must already be the flattened,
    mask-selected sample arrays (1-D)."""
    sample_count = int(lr_values.size)
    if sample_count == 0:
        return BandDiscrepancy(band_name, ComputationStatus.NOT_COMPUTABLE, valid_pixel_count, 0, None, None, None, None)

    errors = sr_down_values.astype(np.float64) - lr_values.astype(np.float64)
    mean_abs, rmse, max_abs, normalized_rmse = _stats(errors, lr_values.astype(np.float64))
    return BandDiscrepancy(
        band_name, ComputationStatus.COMPUTABLE, valid_pixel_count, sample_count, mean_abs, rmse, max_abs, normalized_rmse
    )


def _validate_inputs(
    lr: torch.Tensor, sr: torch.Tensor, mask: np.ndarray, band_names: Sequence[str], scale_factor: int
) -> None:
    if lr.ndim != 3:
        raise ShapeMismatchError(f"Expected LR tensor shape (bands, H, W), got {tuple(lr.shape)}.")
    if sr.ndim != 3:
        raise ShapeMismatchError(f"Expected SR tensor shape (bands, H, W), got {tuple(sr.shape)}.")
    if lr.shape[0] != sr.shape[0]:
        raise ShapeMismatchError(
            f"LR has {lr.shape[0]} band(s) but SR has {sr.shape[0]} band(s) -- must match."
        )
    if len(band_names) != lr.shape[0]:
        raise ShapeMismatchError(
            f"band_names has {len(band_names)} entries but the tensors have {lr.shape[0]} band(s)."
        )
    expected_sr_shape = (lr.shape[1] * scale_factor, lr.shape[2] * scale_factor)
    if tuple(sr.shape[1:]) != expected_sr_shape:
        raise ShapeMismatchError(
            f"SR spatial shape {tuple(sr.shape[1:])} does not equal LR shape {tuple(lr.shape[1:])} "
            f"times scale_factor={scale_factor} (expected {expected_sr_shape})."
        )
    mask_arr = np.asarray(mask)
    if mask_arr.shape != tuple(lr.shape[1:]):
        raise InvalidMaskError(
            f"Mask shape {mask_arr.shape} does not match the LR grid shape {tuple(lr.shape[1:])}."
        )


def compute_downsample_consistency(
    lr: torch.Tensor,
    sr: torch.Tensor,
    mask: np.ndarray,
    *,
    band_names: Sequence[str],
    scale_factor: int,
) -> DownsampleConsistencyResult:
    """Downsample ``sr`` back to the LR grid and compare it against ``lr``.

    Args:
        lr: Original LR input tensor, shape (bands, H, W).
        sr: SR output tensor, shape (bands, H*scale_factor, W*scale_factor).
        mask: Boolean validity mask, shape (H, W) -- from
            `frame.preprocessing.ValidityMask.array`. Masked (False) pixels
            are excluded from every statistic below; a pixel is never
            treated as a real zero-reflectance observation just because it
            was zero-filled upstream.
        band_names: Name of each band along axis 0, in order.
        scale_factor: The SR model's upscaling factor.

    Returns:
        A DownsampleConsistencyResult with an overall (all bands pooled) and
        a per-band BandDiscrepancy.
    """
    _validate_inputs(lr, sr, mask, band_names, scale_factor)

    sr_down = downsample_to_lr_grid(sr, scale_factor)

    lr_np = lr.detach().cpu().numpy()
    sr_down_np = sr_down.detach().cpu().numpy()
    mask_np = np.asarray(mask, dtype=bool)

    return downsample_consistency_from_arrays(
        lr_np, sr_down_np, mask_np, band_names=band_names, scale_factor=int(scale_factor)
    )


def downsample_consistency_from_arrays(
    lr_np: np.ndarray,
    sr_down_np: np.ndarray,
    mask_np: np.ndarray,
    *,
    band_names: Sequence[str],
    scale_factor: int,
) -> DownsampleConsistencyResult:
    """Same computation as `compute_downsample_consistency`, but taking an
    *already*-downsampled SR array (numpy, on the LR grid) directly.

    Exists so a caller that also needs the downsampled-SR array for another
    purpose (e.g. `frame.consistency.report.run_consistency_diagnostics`,
    which reuses it for the spectral-ratio comparisons) does not have to pay
    for -- or risk any inconsistency from -- downsampling the same SR tensor
    twice.
    """
    valid_pixel_count = int(mask_np.sum())
    total_pixel_count = int(mask_np.size)
    mask_coverage = float(mask_np.mean())

    per_band: Dict[str, BandDiscrepancy] = {}
    for i, name in enumerate(band_names):
        per_band[name] = _band_discrepancy(
            name, lr_np[i][mask_np], sr_down_np[i][mask_np], valid_pixel_count
        )

    # Overall: pool every band's valid samples together (see BandDiscrepancy
    # docstring for the valid_pixel_count vs. sample_count distinction).
    if valid_pixel_count == 0:
        pooled_lr = np.empty(0)
        pooled_sr = np.empty(0)
    else:
        pooled_lr = np.concatenate([lr_np[i][mask_np] for i in range(len(band_names))])
        pooled_sr = np.concatenate([sr_down_np[i][mask_np] for i in range(len(band_names))])
    overall = _band_discrepancy(None, pooled_lr, pooled_sr, valid_pixel_count)

    return DownsampleConsistencyResult(
        downsample_method=AREA_AVERAGE_POOL,
        scale_factor=int(scale_factor),
        band_names=tuple(band_names),
        valid_pixel_count=valid_pixel_count,
        total_pixel_count=total_pixel_count,
        mask_coverage=mask_coverage,
        overall=overall,
        per_band=per_band,
    )
