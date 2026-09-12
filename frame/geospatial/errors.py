"""Exception types raised by the FRAME geospatial layer.

The central rule this layer enforces (docs/FRAME_TECHNICAL_SPEC.md Section
10): geospatial context is never fabricated. If a CRS or affine transform
is required and absent, that is a hard, clearly-named error -- never a
silently-applied default.
"""


class GeospatialError(Exception):
    """Base class for every error raised by frame.geospatial."""


class MissingCRSError(GeospatialError):
    """Raised when a CRS is required but not present -- never fabricated."""


class MissingTransformError(GeospatialError):
    """Raised when an affine transform is required but not present."""


class InconsistentMetadataError(GeospatialError):
    """Raised when derived/read-back geospatial values don't agree with
    what was expected (e.g. a recomputed footprint drifting from the
    declared bounds, or a GeoTIFF's band count not matching its band
    names)."""
