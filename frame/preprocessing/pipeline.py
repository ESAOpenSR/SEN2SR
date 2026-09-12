"""Top-level FRAME preprocessing entry point for the proven RGBN path.

``preprocess_rgbn`` turns a raw Sentinel-2 L2A band stack into exactly the
tensor shape/dtype/band-order that ``SEN2SRLite/NonReference_RGBN_x4``
expects (as used, unmodified, by experiments/baseline/run_baseline.py),
while keeping a separate validity mask and a typed metadata record
alongside it.

This module never imports anything from ``sen2sr`` and never calls the
model -- it only prepares the model's input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from frame.preprocessing.masks import ValidityMask
from frame.preprocessing.metadata import RasterMetadata, validate_metadata_matches_array
from frame.preprocessing.reflectance import to_reflectance
from frame.preprocessing.validation import reorder_bands, validate_bands, validate_resolution, validate_shape

# The exact band set/order sen2sr's NonReference_RGBN_x4 path expects, per
# the upstream README and experiments/baseline/run_baseline.py: Red, Green,
# Blue, NIR.
RGBN_BANDS: Tuple[str, str, str, str] = ("B04", "B03", "B02", "B08")
RGBN_RESOLUTION_M: float = 10.0
RGBN_SR_VARIANT = "SEN2SRLite/NonReference_RGBN_x4"

# The model's proven native patch size (experiments/baseline/run_baseline.py
# EDGE_SIZE_PX). Not an architectural fact we've verified beyond that patch
# size -- see docs/FRAME_TECHNICAL_SPEC.md Section 7 -- so it is a default,
# overridable via patch_size=None or a different value.
PROVEN_PATCH_SIZE = 128


@dataclass(frozen=True)
class PreprocessedInput:
    """The validated, model-ready result of preprocess_rgbn."""

    tensor: torch.Tensor
    mask: ValidityMask
    metadata: RasterMetadata


def preprocess_rgbn(
    array: np.ndarray,
    *,
    band_names: Sequence[str],
    input_scale: str,
    resolution_m: float,
    nodata_value: Optional[float] = None,
    scl: Optional[np.ndarray] = None,
    scl_invalid_classes: Optional[frozenset] = None,
    patch_size: Optional[int] = PROVEN_PATCH_SIZE,
    crs: Optional[str] = None,
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    acquisition_timestamp: Optional[str] = None,
    require_geospatial: bool = False,
) -> PreprocessedInput:
    """Validate and convert a raw RGBN band stack into model-ready form.

    Args:
        array: Raw band stack, shape (4, H, W), any order (see band_names).
        band_names: The band each entry of axis 0 of ``array`` actually is,
            e.g. ``["B08", "B02", "B04", "B03"]`` -- reordered internally to
            ``RGBN_BANDS``.
        input_scale: ``"raw_digital_number"`` or ``"reflectance"`` -- see
            frame.preprocessing.reflectance.to_reflectance. No guessing.
        resolution_m: The native resolution of ``array``; must equal
            ``RGBN_RESOLUTION_M`` (10.0) or a clear error is raised.
        nodata_value: If given, pixels where every band equals this value
            are flagged invalid in the returned mask.
        scl: Optional Sentinel-2 L2A Scene Classification band, shape
            (H, W); cloud/shadow/etc. pixels are flagged invalid.
        scl_invalid_classes: Override for which SCL codes count as invalid
            (defaults to the standard ESA convention).
        patch_size: Required square patch size (defaults to the proven
            128x128 patch used by Baseline 0). Pass ``None`` to skip this
            check (for future tiling-orchestration phases).
        crs, transform, bounds, acquisition_timestamp: Optional geospatial/
            provenance context to preserve in the output metadata.
        require_geospatial: If True, raise ``MissingMetadataError`` when
            ``crs``/``transform`` are not supplied.

    Returns:
        A PreprocessedInput with a (4, H, W) float32 torch.Tensor in
        RGBN_BANDS order, a co-sized ValidityMask, and a RasterMetadata
        record.
    """
    validate_shape(array, expected_size=patch_size)
    validate_bands(list(band_names), list(RGBN_BANDS))
    validate_resolution(resolution_m, expected_resolution_m=RGBN_RESOLUTION_M)

    reordered = reorder_bands(array, list(band_names), list(RGBN_BANDS))

    mask = ValidityMask.all_valid(shape=reordered.shape[1:])
    if nodata_value is not None:
        mask = mask.combine(ValidityMask.from_nodata(reordered, nodata_value=nodata_value))
    if scl is not None:
        mask = mask.combine(ValidityMask.from_scl(scl, invalid_classes=scl_invalid_classes))

    reflectance = to_reflectance(reordered, input_scale=input_scale)
    tensor = torch.from_numpy(reflectance)

    metadata = RasterMetadata(
        crs=crs,
        transform=transform,
        bounds=bounds,
        resolution_m=resolution_m,
        width=reordered.shape[2],
        height=reordered.shape[1],
        band_names=RGBN_BANDS,
        acquisition_timestamp=acquisition_timestamp,
        nodata_value=nodata_value,
        cloud_mask_coverage=mask.coverage(),
        sr_variant=RGBN_SR_VARIANT,
    )
    validate_metadata_matches_array(metadata, reordered)
    if require_geospatial:
        metadata.require_geospatial()

    return PreprocessedInput(tensor=tensor, mask=mask, metadata=metadata)
