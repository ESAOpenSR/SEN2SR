"""Deterministic output-transform/metadata derivation for the SR pipeline.

docs/FRAME_TECHNICAL_SPEC.md Section 10 defines the invariant this module
implements: increasing pixel density via SR changes pixel size, not
projection or footprint. Concretely, for a GDAL-style affine 6-tuple
``(a, b, c, d, e, f)``:

    - a, b, d, e (the transform's linear part: pixel size + any
      rotation/shear) all divide by the SR scale factor.
    - c, f (the origin -- the world coordinate of the upper-left corner)
      are unchanged.
    - The CRS is unchanged (SR does not reproject).
    - The bounds (footprint) are unchanged, because scaling the linear part
      by 1/N while also scaling width/height by N leaves the mapped extent
      of the pixel grid identical -- this is verified, not just asserted,
      by recomputing bounds from the output transform/dimensions and
      comparing them against the input bounds.

This module never fabricates a missing CRS or transform -- see
frame.geospatial.metadata.validate_geospatial_completeness, called before
any derivation proceeds.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from frame.geospatial.errors import InconsistentMetadataError
from frame.geospatial.metadata import bounds_from_transform, validate_geospatial_completeness
from frame.preprocessing.metadata import RasterMetadata

# The proven path: 10 m RGBN bands -> 2.5 m, a 4x scale factor (see
# experiments/baseline/README.md and frame/preprocessing/pipeline.py's
# RGBN_BANDS/RGBN_RESOLUTION_M). Support this first; nothing here assumes
# it is the ONLY supported factor -- scale_factor is always an explicit
# argument.
RGBN_SCALE_FACTOR = 4

# Floating-point-only tolerance for the footprint self-consistency check
# below -- not a scientific threshold, just headroom for float64 rounding.
_FOOTPRINT_CONSISTENCY_TOLERANCE = 1e-6


def derive_output_transform(
    transform: Sequence[float], scale_factor: float
) -> Tuple[float, float, float, float, float, float]:
    """Scale an affine transform's linear part by 1/scale_factor, keeping
    the origin (c, f) fixed."""
    a, b, c, d, e, f = transform
    return (a / scale_factor, b / scale_factor, c, d / scale_factor, e / scale_factor, f)


def derive_output_metadata(
    input_metadata: RasterMetadata,
    *,
    scale_factor: float,
    output_band_names: Sequence[str],
) -> RasterMetadata:
    """Derive the output RasterMetadata for an SR pass at ``scale_factor``.

    Requires ``input_metadata.crs`` and ``input_metadata.transform`` to be
    present (raises MissingCRSError / MissingTransformError otherwise --
    never fabricated). Output width/height are the input's times
    ``scale_factor``; output resolution_m is the input's divided by it.
    """
    validate_geospatial_completeness(input_metadata)

    output_width = round(input_metadata.width * scale_factor)
    output_height = round(input_metadata.height * scale_factor)
    output_transform = derive_output_transform(input_metadata.transform, scale_factor)
    output_bounds = bounds_from_transform(output_transform, output_width, output_height)

    if input_metadata.bounds is not None:
        drift = max(
            abs(a - b) for a, b in zip(output_bounds, input_metadata.bounds)
        )
        if drift > _FOOTPRINT_CONSISTENCY_TOLERANCE:
            raise InconsistentMetadataError(
                f"Derived output footprint {output_bounds} does not match the input "
                f"footprint {input_metadata.bounds} (max drift {drift}); refusing to "
                "silently proceed with an inconsistent geographic extent."
            )

    return RasterMetadata(
        crs=input_metadata.crs,
        transform=output_transform,
        bounds=output_bounds,
        resolution_m=input_metadata.resolution_m / scale_factor,
        width=output_width,
        height=output_height,
        band_names=tuple(output_band_names),
        acquisition_timestamp=input_metadata.acquisition_timestamp,
        nodata_value=input_metadata.nodata_value,
        cloud_mask_coverage=input_metadata.cloud_mask_coverage,
        sr_variant=input_metadata.sr_variant,
    )
