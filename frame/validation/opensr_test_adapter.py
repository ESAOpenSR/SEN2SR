"""Dataset adapter for the `opensr-test` benchmark (docs/FRAME_TECHNICAL_SPEC.md
Section 11.B, resolved via the Phase 3.5 validation-resource research).

`load_subset` is the only function here that touches the network (via
`opensr_test.load`, which downloads a pickle from Hugging Face). Everything
else -- `extract_sample` and the facts it depends on -- is pure, offline
logic, unit-tested against a synthetic dict shaped exactly like a real
loaded subset (see frame/tests/test_validation_adapter.py).

Facts this module relies on, and where they were verified
-----------------------------------------------------------
- A loaded `opensr_test.load(subset)` result is a plain `dict` (NOT the
  `torch.Tensor` its own docstring claims) with keys `L2A`, `L1C`, `HR`,
  `HRharm`, `metadata`, plus two undocumented lowercase duplicates (`hr`,
  `hr_harm`) whose relationship to `HR`/`HRharm` is unexplained upstream and
  which this adapter deliberately does not use -- verified empirically by
  loading the real `spot` subset (opensr-test 1.3.3) during Phase 4
  development.
- `L2A` is (N, 12, H, W) uint16, `L1C` is (N, 13, H, W) uint16 (the HF
  dataset card's own README bullet says "L1C (12 bands)", which is simply
  wrong -- its own detailed band table, and the real array's shape, both
  say 13). `HR`/`HRharm` are (N, 4, H*scale, W*scale) uint16.
- The L2A band order (`L2A_BAND_ORDER` below) is copied verbatim from the
  "L2A Index" column of the dataset's own published band table at
  https://huggingface.co/datasets/isp-uv-es/opensr-test/raw/main/README.md
  -- an authoritative, upstream-published source, not inferred from pixel
  correlation (which was tried during research and found too confounded by
  cross-band brightness correlation in real imagery to be a reliable
  discriminator on its own).
- `HR`/`HRharm`'s 4 bands are described as "RGBNIR" by the same README --
  read here as [Red, Green, Blue, NIR] in that literal order, matching
  `frame.preprocessing.RGBN_BANDS` exactly. This is a naming-convention
  inference, not a table-confirmed fact like the L2A order above --
  flagged as weaker evidence in frame/validation/README.md.
- All four arrays (L2A, L1C, HR, HRharm) are raw Sentinel-2-style digital
  numbers scaled by 10,000, per the dataset README's own literal usage
  example (`... / 10000`) -- identical to
  `frame.preprocessing.reflectance.SENTINEL2_L2A_REFLECTANCE_SCALE`.
- The per-sample `metadata` DataFrame's `affine` column is a comma-separated
  GDAL-style 6-tuple string describing the **HR/SR grid** (its pixel size
  matches the subset's HR resolution, e.g. 2.5 m) -- confirmed against a
  real sample. The LR-grid transform is not given directly and is derived
  here by scaling the HR transform's linear terms by `scale_factor`
  (exact inverse of `frame.geospatial.transform.derive_output_transform`'s
  documented invariant), keeping the origin unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from frame.preprocessing import RGBN_BANDS
from frame.preprocessing.metadata import RasterMetadata
from frame.preprocessing.reflectance import to_reflectance
from frame.preprocessing.validation import reorder_bands
from frame.validation.errors import (
    BenchmarkFormatError,
    SampleIndexError,
    UnsupportedSubsetError,
)

# Only the subsets whose LR grid is exactly 128x128 (our proven patch size)
# and whose scale factor is x4 (our proven model's scale) are supported in
# Phase 4 -- per the Phase 3.5 research report and this phase's explicit
# scope. `naip` (121x121 LR) and `venus` (x2 scale) are deliberately excluded,
# not silently accepted with a mismatched shape.
SUPPORTED_SUBSETS: Tuple[str, ...] = ("spot", "spain_crops", "spain_urban")

EXPECTED_LR_SIZE = 128
EXPECTED_SCALE_FACTOR = 4

# Verbatim from the "L2A Index" column of
# https://huggingface.co/datasets/isp-uv-es/opensr-test/raw/main/README.md
# (accessed during Phase 4 development; opensr-test HF dataset repo commit
# e4600b9c74a621adeec047e5f6cc7a2d70a58134). L2A has 12 bands (L1C's 13,
# minus B10, the standard ESA L2A convention).
L2A_BAND_ORDER: Tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12",
)
EXPECTED_L2A_BAND_COUNT = len(L2A_BAND_ORDER)  # 12

L2A_BAND_ORDER_SOURCE = (
    "https://huggingface.co/datasets/isp-uv-es/opensr-test/raw/main/README.md "
    "(\"L2A Index\" column of the published band table), HF repo commit "
    "e4600b9c74a621adeec047e5f6cc7a2d70a58134, accessed 2026 during Phase 4 development."
)

# The exact 4 bands and order our proven SEN2SRLite/NonReference_RGBN_x4
# path requires -- identical to frame.preprocessing.RGBN_BANDS.
RGBN_BAND_NAMES: Tuple[str, str, str, str] = RGBN_BANDS

DEFAULT_HR_VARIANT = "HRharm"


@dataclass(frozen=True)
class OpenSRTestSample:
    """One evaluation-ready sample extracted from a loaded opensr-test subset."""

    subset: str
    sample_index: int
    roi_id: Optional[str]
    lr_reflectance: torch.Tensor  # (4, H, W) float32, RGBN_BAND_NAMES order, /10000-scaled
    hr_reflectance: torch.Tensor  # (4, H*scale, W*scale) float32, same band order
    hr_variant: str  # "HR" or "HRharm" -- which key hr_reflectance came from
    scale_factor: int
    lr_metadata: RasterMetadata
    hr_metadata: RasterMetadata
    dataset_version: Optional[str]
    l2a_band_order_source: str


def load_subset(subset: str, *, version: str = "v3", cache_dir: Optional[str] = None):
    """Download (or reuse the local cache of) one opensr-test subset.

    Thin wrapper around `opensr_test.load` -- the only network-touching
    function in this module. Raises `UnsupportedSubsetError` before
    attempting any download if `subset` is outside `SUPPORTED_SUBSETS`.
    """
    if subset not in SUPPORTED_SUBSETS:
        raise UnsupportedSubsetError(
            f"Subset {subset!r} is not supported in Phase 4. Supported: {SUPPORTED_SUBSETS}."
        )
    import opensr_test  # imported lazily: frame.validation must stay importable

    kwargs: Dict[str, Any] = {"version": version}
    if cache_dir is not None:
        kwargs["model_dir"] = cache_dir
    return opensr_test.load(subset, **kwargs)


def _parse_affine(affine_str: str) -> Tuple[float, float, float, float, float, float]:
    parts = [float(p) for p in affine_str.split(",")]
    if len(parts) != 6:
        raise BenchmarkFormatError(
            f"Expected a 6-value affine string, got {len(parts)} value(s): {affine_str!r}."
        )
    return tuple(parts)  # type: ignore[return-value]


def _derive_lr_transform(
    hr_transform: Tuple[float, float, float, float, float, float], scale_factor: int
) -> Tuple[float, float, float, float, float, float]:
    """LR pixel size = HR pixel size * scale_factor; origin (c, f) unchanged.

    Exact inverse of frame.geospatial.transform.derive_output_transform's
    documented invariant (which divides by scale_factor going LR -> SR);
    here we multiply, going HR/SR -> LR.
    """
    a, b, c, d, e, f = hr_transform
    return (a * scale_factor, b * scale_factor, c, d * scale_factor, e * scale_factor, f)


def extract_sample(
    loaded: Dict[str, Any],
    *,
    subset: str,
    sample_index: int,
    hr_variant: str = DEFAULT_HR_VARIANT,
    dataset_version: Optional[str] = None,
) -> OpenSRTestSample:
    """Extract one model-ready, geospatially-described sample from a loaded subset.

    Args:
        loaded: The dict returned by `load_subset`/`opensr_test.load`.
        subset: Name of the subset `loaded` came from (recorded, and
            validated against `SUPPORTED_SUBSETS`).
        sample_index: Which sample (0-indexed) within the subset to extract.
        hr_variant: `"HRharm"` (default, matching the upstream README's own
            usage) or `"HR"` (unharmonized).
        dataset_version: Recorded as-is for reproducibility (e.g.
            `opensr_test.__version__`); not validated.

    Returns:
        An OpenSRTestSample with the 4-band RGBN LR/HR tensors, reflectance-
        scaled, plus geospatial metadata for both grids.
    """
    if subset not in SUPPORTED_SUBSETS:
        raise UnsupportedSubsetError(
            f"Subset {subset!r} is not supported in Phase 4. Supported: {SUPPORTED_SUBSETS}."
        )
    if hr_variant not in ("HR", "HRharm"):
        raise BenchmarkFormatError(f"hr_variant must be 'HR' or 'HRharm', got {hr_variant!r}.")

    l2a = loaded["L2A"]
    hr_stack = loaded[hr_variant]
    metadata_df = loaded["metadata"]

    n = l2a.shape[0]
    if not (0 <= sample_index < n):
        raise SampleIndexError(f"sample_index {sample_index} out of range for subset with {n} sample(s).")

    if l2a.shape[1] != EXPECTED_L2A_BAND_COUNT:
        raise BenchmarkFormatError(
            f"Expected L2A to have {EXPECTED_L2A_BAND_COUNT} bands (per the documented band table), "
            f"got {l2a.shape[1]}. The upstream dataset format may have changed."
        )

    lr_raw = l2a[sample_index]  # (12, H, W) uint16
    hr_raw = hr_stack[sample_index]  # (4, H*scale, W*scale) uint16

    h_lr = lr_raw.shape[-1]
    h_hr = hr_raw.shape[-1]
    if h_hr % h_lr != 0:
        raise BenchmarkFormatError(
            f"HR size {h_hr} is not an integer multiple of LR size {h_lr}; cannot infer a scale factor."
        )
    scale_factor = h_hr // h_lr
    if hr_raw.shape != (4, h_lr * scale_factor, h_lr * scale_factor) or lr_raw.shape[-2] != lr_raw.shape[-1]:
        raise BenchmarkFormatError(
            f"HR shape {hr_raw.shape} is not consistent with a square LR grid {lr_raw.shape[-2:]} "
            f"scaled by an integer factor."
        )

    # Select and reorder the 4 RGBN bands out of the full L2A stack, using
    # the documented band-order table (reuses frame.preprocessing's own
    # reorder_bands rather than re-deriving index arithmetic here).
    lr_rgbn_raw = reorder_bands(lr_raw, list(L2A_BAND_ORDER), list(RGBN_BAND_NAMES))
    lr_reflectance = torch.from_numpy(to_reflectance(lr_rgbn_raw, input_scale="raw_digital_number"))

    # HR/HRharm are already RGBNIR order == RGBN_BAND_NAMES order (see module
    # docstring) -- no reordering, only reflectance scaling.
    hr_reflectance = torch.from_numpy(to_reflectance(hr_raw, input_scale="raw_digital_number"))

    row = metadata_df.iloc[sample_index]
    roi_id = str(row["roi"]) if "roi" in metadata_df.columns else None
    crs = str(row["crs"]) if "crs" in metadata_df.columns else None
    hr_transform = _parse_affine(str(row["affine"])) if "affine" in metadata_df.columns else None

    if crs is not None and hr_transform is not None:
        hr_resolution_m = abs(hr_transform[0])
        hr_metadata = RasterMetadata(
            crs=crs,
            transform=hr_transform,
            bounds=None,
            resolution_m=hr_resolution_m,
            width=hr_reflectance.shape[-1],
            height=hr_reflectance.shape[-2],
            band_names=RGBN_BAND_NAMES,
            acquisition_timestamp=None,
            nodata_value=None,
            cloud_mask_coverage=None,
            sr_variant=None,
        )
        lr_transform = _derive_lr_transform(hr_transform, scale_factor)
        lr_metadata = RasterMetadata(
            crs=crs,
            transform=lr_transform,
            bounds=None,
            resolution_m=hr_resolution_m * scale_factor,
            width=lr_reflectance.shape[-1],
            height=lr_reflectance.shape[-2],
            band_names=RGBN_BAND_NAMES,
            acquisition_timestamp=None,
            nodata_value=None,
            cloud_mask_coverage=None,
            sr_variant=None,
        )
    else:
        hr_metadata = RasterMetadata.unknown(
            band_names=RGBN_BAND_NAMES,
            width=hr_reflectance.shape[-1],
            height=hr_reflectance.shape[-2],
            resolution_m=float("nan"),
            sr_variant="n/a",
        )
        lr_metadata = RasterMetadata.unknown(
            band_names=RGBN_BAND_NAMES,
            width=lr_reflectance.shape[-1],
            height=lr_reflectance.shape[-2],
            resolution_m=float("nan"),
            sr_variant="n/a",
        )

    return OpenSRTestSample(
        subset=subset,
        sample_index=sample_index,
        roi_id=roi_id,
        lr_reflectance=lr_reflectance,
        hr_reflectance=hr_reflectance,
        hr_variant=hr_variant,
        scale_factor=scale_factor,
        lr_metadata=lr_metadata,
        hr_metadata=hr_metadata,
        dataset_version=dataset_version,
        l2a_band_order_source=L2A_BAND_ORDER_SOURCE,
    )
