"""Exception types raised by the FRAME preprocessing layer.

Every rejection of an unsupported input raises one of these, with a message
naming exactly what was wrong -- callers should never have to guess why an
input was rejected.
"""


class PreprocessingError(Exception):
    """Base class for every error raised by frame.preprocessing."""


class InvalidShapeError(PreprocessingError):
    """Raised when an input array's dimensionality or shape is unsupported."""


class UnsupportedBandsError(PreprocessingError):
    """Raised when the band set/order of an input does not match what is required."""


class UnsupportedResolutionError(PreprocessingError):
    """Raised when an input's spatial resolution does not match what is required."""


class InvalidInputScaleError(PreprocessingError):
    """Raised when an unrecognized reflectance input-scale identifier is given."""


class MissingMetadataError(PreprocessingError):
    """Raised when geospatial metadata is required but not available."""
