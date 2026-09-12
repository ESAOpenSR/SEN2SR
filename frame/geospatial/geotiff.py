"""GeoTIFF read/write for the FRAME geospatial layer.

This is the one module in `frame.geospatial` that genuinely needs
`rasterio`: writing a spec-compliant GeoTIFF (embedded CRS, affine
transform/geotransform, per-band descriptions, nodata value, custom tags)
and reading one back for independent verification is exactly what a raster
I/O library is for -- reimplementing the GeoTIFF format's tag/IFD structure
by hand would be reinventing GDAL. The dependency is confined to this file;
the rest of `frame.geospatial` (metadata.py, transform.py) has no rasterio
import.

Float32 reflectance values are written directly (no rescaling to
integer digital numbers) -- see docs/FRAME_TECHNICAL_SPEC.md Section 6's
open question on eventual float32-vs-uint16 export; float32 is the simpler,
lossless choice for this phase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import rasterio
from rasterio.transform import Affine

from frame.geospatial.errors import MissingCRSError
from frame.geospatial.metadata import validate_geospatial_completeness
from frame.preprocessing.metadata import RasterMetadata

# Custom GeoTIFF tag names used to round-trip FRAME-specific provenance
# fields that GeoTIFF has no dedicated slot for.
_TAG_SR_VARIANT = "FRAME_SR_VARIANT"
_TAG_ACQUISITION_TIMESTAMP = "FRAME_SOURCE_ACQUISITION_TIMESTAMP"
_TAG_CLOUD_MASK_COVERAGE = "FRAME_CLOUD_MASK_COVERAGE"
_TAG_PIPELINE = "FRAME_PIPELINE"
_PIPELINE_TAG_VALUE = "FRAME geospatial export (Phase 2)"


def write_geotiff(path: Union[str, Path], array: np.ndarray, metadata: RasterMetadata) -> None:
    """Write ``array`` (bands, H, W) as a GeoTIFF using ``metadata``.

    Requires ``metadata.crs`` and ``metadata.transform`` to be present
    (raises MissingCRSError / MissingTransformError otherwise -- a missing
    CRS is never fabricated as a default). Band order in the file matches
    ``metadata.band_names`` order exactly (band i's description is
    ``band_names[i]``).
    """
    validate_geospatial_completeness(metadata)

    array = np.asarray(array, dtype="float32")
    bands, height, width = array.shape
    if bands != len(metadata.band_names) or height != metadata.height or width != metadata.width:
        raise ValueError(
            f"Array shape {array.shape} does not match metadata "
            f"(bands={len(metadata.band_names)}, height={metadata.height}, width={metadata.width})."
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=bands,
        dtype="float32",
        crs=metadata.crs,
        transform=Affine(*metadata.transform),
        nodata=metadata.nodata_value,
    ) as dst:
        dst.write(array)
        for i, name in enumerate(metadata.band_names, start=1):
            dst.set_band_description(i, name)

        tags = {_TAG_PIPELINE: _PIPELINE_TAG_VALUE}
        if metadata.sr_variant is not None:
            tags[_TAG_SR_VARIANT] = metadata.sr_variant
        if metadata.acquisition_timestamp is not None:
            tags[_TAG_ACQUISITION_TIMESTAMP] = metadata.acquisition_timestamp
        if metadata.cloud_mask_coverage is not None:
            tags[_TAG_CLOUD_MASK_COVERAGE] = repr(metadata.cloud_mask_coverage)
        dst.update_tags(**tags)


def read_geotiff(
    path: Union[str, Path], *, require_crs: bool = True
) -> Tuple[np.ndarray, RasterMetadata]:
    """Read a GeoTIFF back into (array, RasterMetadata) for independent
    verification of what was actually written to disk.

    Args:
        path: GeoTIFF file to read.
        require_crs: If True (default), raise MissingCRSError when the file
            has no embedded CRS, rather than silently returning
            ``crs=None`` -- matching this platform's stance that
            geospatial validity is not optional. Pass False to inspect a
            file that may legitimately lack one.
    """
    with rasterio.open(path) as src:
        array = src.read()
        crs = src.crs.to_string() if src.crs is not None else None

        if require_crs and crs is None:
            raise MissingCRSError(f"{path} has no embedded CRS.")

        transform: Tuple[float, float, float, float, float, float] = tuple(src.transform)[:6]
        bounds = tuple(src.bounds)  # (left, bottom, right, top) == (minx, miny, maxx, maxy)

        if src.descriptions and all(src.descriptions):
            band_names = tuple(src.descriptions)
        else:
            band_names = tuple(f"band_{i}" for i in range(1, src.count + 1))

        tags = src.tags()
        cloud_mask_coverage = tags.get(_TAG_CLOUD_MASK_COVERAGE)

        metadata = RasterMetadata(
            crs=crs,
            transform=transform,
            bounds=bounds,
            resolution_m=abs(transform[0]),
            width=src.width,
            height=src.height,
            band_names=band_names,
            acquisition_timestamp=tags.get(_TAG_ACQUISITION_TIMESTAMP),
            nodata_value=src.nodata,
            cloud_mask_coverage=float(cloud_mask_coverage) if cloud_mask_coverage is not None else None,
            sr_variant=tags.get(_TAG_SR_VARIANT),
        )

    return array, metadata
