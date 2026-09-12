"""The FRAME geospatial layer's raster/metadata representation.

This layer deliberately reuses ``frame.preprocessing.metadata.RasterMetadata``
rather than defining a second, overlapping dataclass: it already carries
every field this phase needs (CRS, affine transform, bounds, resolution,
width, height, band names, nodata value), plus the provenance fields
(``acquisition_timestamp``, ``sr_variant``, ``cloud_mask_coverage``) needed
for the GeoTIFF tags in Section 4 of the Phase 2 requirements. Splitting
that into two near-identical representations would create two competing
sources of truth for the same raster's geospatial identity -- so this
module re-exports it and adds the geospatial-layer-specific operations on
top: footprint computation and the "never fabricate a missing CRS" gate.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from rasterio.transform import Affine

from frame.geospatial.errors import MissingCRSError, MissingTransformError
from frame.preprocessing.metadata import RasterMetadata

__all__ = ["RasterMetadata", "bounds_from_transform", "validate_geospatial_completeness"]


def bounds_from_transform(
    transform: Sequence[float], width: int, height: int
) -> Tuple[float, float, float, float]:
    """Compute (minx, miny, maxx, maxy) by mapping all four pixel-grid
    corners through the affine transform and taking their bounding box.

    Mapping all four corners (rather than assuming an axis-aligned,
    north-up grid) keeps this correct for a transform with rotation/shear
    terms (b, d non-zero), not just the simple case.
    """
    aff = Affine(*transform)
    corners = [aff @ (0, 0), aff @ (width, 0), aff @ (0, height), aff @ (width, height)]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def validate_geospatial_completeness(metadata: RasterMetadata) -> None:
    """Raise a specific, clear error if CRS or transform is missing.

    Never fabricates a default -- e.g. never substitutes EPSG:4326 for a
    missing CRS. Callers that need geospatial context must fail loudly
    here rather than proceed with an invented value.
    """
    if metadata.crs is None:
        raise MissingCRSError(
            "No CRS is present in this raster's metadata. Refusing to proceed: "
            "a CRS must be supplied by the source data, never fabricated."
        )
    if metadata.transform is None:
        raise MissingTransformError(
            "No affine transform is present in this raster's metadata. Refusing "
            "to proceed: a transform must be supplied by the source data, never fabricated."
        )
