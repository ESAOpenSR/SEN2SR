"""NDVI computation for the FRAME downstream-analysis layer (Phase 6).

NDVI = (NIR - Red) / (NIR + Red). The safe-division core is REUSED from
`frame.consistency.spectral_ratios.compute_ndvi` (Phase 3, already tested)
rather than reimplemented -- this module only adds the ergonomics Phase 6
needs on top: locating the Red/NIR bands by name in a (bands, H, W) stack,
combining the resulting per-pixel computability mask with an optional
external validity mask (e.g. nodata/cloud), and recording enough metadata
(resolution, a grid label) to keep the native-10m and SR-2.5m results from
being confused with each other downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from frame.analysis.errors import MissingBandError, ShapeMismatchError
from frame.consistency.spectral_ratios import compute_ndvi as _compute_ndvi_core

RED_BAND = "B04"
NIR_BAND = "B08"


@dataclass(frozen=True)
class NDVIResult:
    """NDVI computed from one (bands, H, W) reflectance stack.

    ``ndvi`` holds NaN at pixels where ``valid_mask`` is False -- valid
    pixels are guaranteed finite (see
    frame/tests/test_analysis_indices.py::test_no_nan_or_inf_in_valid_pixels).
    ``resolution_m``/``grid_label`` exist specifically so a native-grid and
    an SR-grid NDVIResult are never silently interchanged.
    """

    ndvi: torch.Tensor  # (H, W) float32
    valid_mask: torch.Tensor  # (H, W) bool
    resolution_m: Optional[float]
    grid_label: Optional[str]


def compute_ndvi_from_stack(
    reflectance: torch.Tensor,
    *,
    band_names: Sequence[str],
    external_mask: Optional[np.ndarray] = None,
    resolution_m: Optional[float] = None,
    grid_label: Optional[str] = None,
) -> NDVIResult:
    """Compute NDVI from a (bands, H, W) reflectance stack.

    Args:
        reflectance: (bands, H, W) tensor.
        band_names: Name of each band along axis 0 -- must include
            ``"B04"`` (Red) and ``"B08"`` (NIR).
        external_mask: Optional (H, W) boolean validity mask (e.g. from
            `frame.preprocessing.ValidityMask` or the SR output's own
            nodata detection). Combined (logical AND) with the pixels
            where NIR+Red is safely non-zero. A pixel invalid in either
            sense is excluded from ``valid_mask``.
        resolution_m, grid_label: Recorded on the result for provenance
            (e.g. ``resolution_m=10.0, grid_label="native_10m"`` vs.
            ``resolution_m=2.5, grid_label="sr_2_5m"``) -- not validated,
            purely descriptive metadata carried through to the report.
    """
    if reflectance.ndim != 3:
        raise ShapeMismatchError(
            f"Expected a 3-D (bands, H, W) tensor, got {reflectance.ndim}-D shape {tuple(reflectance.shape)}."
        )

    try:
        red_index = list(band_names).index(RED_BAND)
        nir_index = list(band_names).index(NIR_BAND)
    except ValueError as exc:
        raise MissingBandError(
            f"NDVI requires bands {RED_BAND!r} and {NIR_BAND!r}; band_names={list(band_names)}."
        ) from exc

    red = reflectance[red_index].detach().cpu().numpy()
    nir = reflectance[nir_index].detach().cpu().numpy()

    ndvi_np, computable = _compute_ndvi_core(nir, red)

    valid = computable
    if external_mask is not None:
        external_mask = np.asarray(external_mask, dtype=bool)
        if external_mask.shape != red.shape:
            raise ShapeMismatchError(
                f"external_mask shape {external_mask.shape} does not match the spatial shape {red.shape}."
            )
        valid = valid & external_mask

    return NDVIResult(
        ndvi=torch.from_numpy(ndvi_np.astype("float32")),
        valid_mask=torch.from_numpy(valid),
        resolution_m=resolution_m,
        grid_label=grid_label,
    )
