"""FRAME geospatial layer.

Preserves and reconstructs geospatial metadata (CRS, affine transform,
bounds) around the tensor-based, unmodified upstream `sen2sr` SR pipeline,
and writes/reads the SR result as a real GeoTIFF.

See docs/FRAME_TECHNICAL_SPEC.md Section 10 for the design this implements,
and README.md in this package for the concrete contract.

Public API:
    RasterMetadata (reused from frame.preprocessing.metadata)
    bounds_from_transform, validate_geospatial_completeness
    derive_output_transform, derive_output_metadata, RGBN_SCALE_FACTOR
    write_geotiff, read_geotiff
    Exceptions: GeospatialError and its subclasses (see .errors)
"""

from frame.geospatial.errors import (
    GeospatialError,
    InconsistentMetadataError,
    MissingCRSError,
    MissingTransformError,
)
from frame.geospatial.geotiff import read_geotiff, write_geotiff
from frame.geospatial.metadata import RasterMetadata, bounds_from_transform, validate_geospatial_completeness
from frame.geospatial.transform import RGBN_SCALE_FACTOR, derive_output_metadata, derive_output_transform

__all__ = [
    "RasterMetadata",
    "bounds_from_transform",
    "validate_geospatial_completeness",
    "derive_output_transform",
    "derive_output_metadata",
    "RGBN_SCALE_FACTOR",
    "write_geotiff",
    "read_geotiff",
    "GeospatialError",
    "MissingCRSError",
    "MissingTransformError",
    "InconsistentMetadataError",
]
