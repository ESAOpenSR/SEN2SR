"""Band-wise spectral shape checks (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).

Compares simple inter-band relationships -- NDVI and the B08/B04 ratio --
between the original LR input and the SR output downsampled back to the LR
grid (see `frame.consistency.downsample`). The point is to catch a case the
plain per-band discrepancy check (`frame.consistency.band_discrepancy`)
would not directly surface: each band individually looking close to the LR
observation while the *relationship between bands* (the spectral "shape")
has still drifted.

Like the rest of this package, this is a self-consistency check against the
real LR observation, not an accuracy metric against ground truth, and it
never converts its output into a pass/fail judgment
(`frame.consistency.status.ComputationStatus` only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from frame.consistency.errors import MissingBandError
from frame.consistency.status import ComputationStatus

# A numerical floor purely to keep division well-defined -- NOT a scientific
# threshold. A pixel whose denominator magnitude falls at or below this is
# marked "not computable for this pixel" (excluded), never silently divided
# into a fabricated near-infinite value.
DIVISION_EPSILON = 1e-6

NDVI = "NDVI"
B08_B04_RATIO = "B08_B04_ratio"


@dataclass(frozen=True)
class SpectralIndexComparison:
    index_name: str
    status: ComputationStatus
    valid_pixel_count: int
    mean_abs_discrepancy: Optional[float]
    rmse: Optional[float]
    max_abs_discrepancy: Optional[float]


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Element-wise NDVI = (NIR - Red) / (NIR + Red).

    Returns (ndvi, computable): ``computable`` is False wherever the
    denominator's magnitude is too small to divide by safely; ``ndvi`` at
    those positions is not a meaningful value and must not be read without
    checking ``computable`` first.
    """
    nir64 = np.asarray(nir, dtype=np.float64)
    red64 = np.asarray(red, dtype=np.float64)
    denom = nir64 + red64
    computable = np.abs(denom) > DIVISION_EPSILON
    ndvi = np.full(denom.shape, np.nan, dtype=np.float64)
    ndvi[computable] = (nir64[computable] - red64[computable]) / denom[computable]
    return ndvi, computable


def compute_simple_ratio(numerator: np.ndarray, denominator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Element-wise ``numerator / denominator``, safe against division by zero.

    Returns (ratio, computable), with the same contract as `compute_ndvi`.
    """
    num64 = np.asarray(numerator, dtype=np.float64)
    denom64 = np.asarray(denominator, dtype=np.float64)
    computable = np.abs(denom64) > DIVISION_EPSILON
    ratio = np.full(denom64.shape, np.nan, dtype=np.float64)
    ratio[computable] = num64[computable] / denom64[computable]
    return ratio, computable


def _band_index(band_names: Sequence[str], band: str) -> int:
    try:
        return list(band_names).index(band)
    except ValueError as exc:
        raise MissingBandError(
            f"Band {band!r} is required for this spectral-index comparison but is not present in "
            f"band_names={list(band_names)}."
        ) from exc


def _compare(
    index_name: str, lr_index: np.ndarray, lr_computable: np.ndarray, sr_index: np.ndarray, sr_computable: np.ndarray, mask: np.ndarray
) -> SpectralIndexComparison:
    combined = np.asarray(mask, dtype=bool) & lr_computable & sr_computable
    n = int(combined.sum())
    if n == 0:
        return SpectralIndexComparison(index_name, ComputationStatus.NOT_COMPUTABLE, 0, None, None, None)

    diff = sr_index[combined] - lr_index[combined]
    abs_diff = np.abs(diff)
    return SpectralIndexComparison(
        index_name,
        ComputationStatus.COMPUTABLE,
        n,
        float(abs_diff.mean()),
        float(np.sqrt(np.mean(diff**2))),
        float(abs_diff.max()),
    )


def compute_ndvi_comparison(
    lr: np.ndarray, sr_downsampled: np.ndarray, mask: np.ndarray, *, band_names: Sequence[str]
) -> SpectralIndexComparison:
    """Compare NDVI computed from ``lr`` against NDVI computed from
    ``sr_downsampled`` (both (bands, H, W) arrays on the same LR grid)."""
    red_i = _band_index(band_names, "B04")
    nir_i = _band_index(band_names, "B08")

    lr_ndvi, lr_computable = compute_ndvi(lr[nir_i], lr[red_i])
    sr_ndvi, sr_computable = compute_ndvi(sr_downsampled[nir_i], sr_downsampled[red_i])

    return _compare(NDVI, lr_ndvi, lr_computable, sr_ndvi, sr_computable, mask)


def compute_b08_b04_ratio_comparison(
    lr: np.ndarray, sr_downsampled: np.ndarray, mask: np.ndarray, *, band_names: Sequence[str]
) -> SpectralIndexComparison:
    """Compare the B08/B04 ratio computed from ``lr`` against the same ratio
    computed from ``sr_downsampled``."""
    red_i = _band_index(band_names, "B04")
    nir_i = _band_index(band_names, "B08")

    lr_ratio, lr_computable = compute_simple_ratio(lr[nir_i], lr[red_i])
    sr_ratio, sr_computable = compute_simple_ratio(sr_downsampled[nir_i], sr_downsampled[red_i])

    return _compare(B08_B04_RATIO, lr_ratio, lr_computable, sr_ratio, sr_computable, mask)
