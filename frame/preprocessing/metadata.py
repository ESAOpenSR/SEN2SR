"""Typed metadata record for a preprocessed input.

RasterMetadata captures everything needed to describe (and, in a later
phase, re-attach as real georeferencing -- see docs/FRAME_TECHNICAL_SPEC.md
Section 10) the provenance and spatial context of a preprocessed array. It
holds plain, library-agnostic types (strings/tuples/floats) rather than
depending on a geospatial library like rasterio/affine/pyproj, since Phase 1
only needs to *preserve* this information, not manipulate it.

The transform, when present, is a GDAL-style affine 6-tuple
``(a, b, c, d, e, f)`` compatible with ``affine.Affine(*transform)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from frame.preprocessing.errors import InvalidShapeError, MissingMetadataError


@dataclass(frozen=True)
class RasterMetadata:
    """Immutable provenance/geospatial record for one preprocessed input."""

    crs: Optional[str]
    transform: Optional[Tuple[float, float, float, float, float, float]]
    bounds: Optional[Tuple[float, float, float, float]]
    resolution_m: float
    width: int
    height: int
    band_names: Tuple[str, ...]
    acquisition_timestamp: Optional[str]
    nodata_value: Optional[float]
    cloud_mask_coverage: Optional[float]
    sr_variant: str

    @classmethod
    def unknown(
        cls,
        *,
        band_names: Sequence[str],
        width: int,
        height: int,
        resolution_m: float,
        sr_variant: str,
    ) -> "RasterMetadata":
        """Build a record with no geospatial context (e.g. a synthetic test array)."""
        return cls(
            crs=None,
            transform=None,
            bounds=None,
            resolution_m=resolution_m,
            width=width,
            height=height,
            band_names=tuple(band_names),
            acquisition_timestamp=None,
            nodata_value=None,
            cloud_mask_coverage=None,
            sr_variant=sr_variant,
        )

    def require_geospatial(self) -> None:
        """Raise MissingMetadataError unless both CRS and transform are present."""
        missing = [name for name, value in (("crs", self.crs), ("transform", self.transform)) if value is None]
        if missing:
            raise MissingMetadataError(
                f"Geospatial metadata required but missing: {missing}."
            )


def validate_metadata_matches_array(metadata: RasterMetadata, array: np.ndarray) -> None:
    """Check that ``metadata``'s declared shape/band-count matches ``array``."""
    if array.ndim != 3:
        raise InvalidShapeError(f"Expected a 3-D (bands, height, width) array, got shape {array.shape}.")

    bands, height, width = array.shape

    if len(metadata.band_names) != bands:
        raise InvalidShapeError(
            f"Metadata declares {len(metadata.band_names)} band(s) {metadata.band_names}, "
            f"but array has {bands} band(s)."
        )
    if metadata.height != height or metadata.width != width:
        raise InvalidShapeError(
            f"Metadata declares {metadata.width}x{metadata.height}, "
            f"but array is {width}x{height}."
        )
