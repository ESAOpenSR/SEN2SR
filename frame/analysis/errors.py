"""Exception types raised by the FRAME downstream-analysis layer.

Same convention as every other `frame.*` package: a structurally malformed
input (a missing required band, a shape mismatch between the native and SR
grids) raises a specific, named error immediately.
"""


class AnalysisError(Exception):
    """Base class for every error raised by frame.analysis."""


class MissingBandError(AnalysisError):
    """Raised when a band required for an index (e.g. B04/B08 for NDVI) is
    not present in the given band_names."""


class ShapeMismatchError(AnalysisError):
    """Raised when tensors handed to an analysis step have incompatible shapes."""
