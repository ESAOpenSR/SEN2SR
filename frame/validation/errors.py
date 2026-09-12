"""Exception types raised by the FRAME reference-based validation layer.

Same convention as `frame.preprocessing.errors`, `frame.geospatial.errors`,
and `frame.consistency.errors`: a structurally malformed input (wrong
subset name, sample index out of range, a shape that doesn't relate the way
it should) raises a specific, named error immediately. A well-formed input
that simply has nothing valid to compute a metric from (e.g. every pixel
masked) is a different situation, reported via
`frame.consistency.status.ComputationStatus` (reused here, not
reimplemented) rather than an exception.
"""


class ValidationError(Exception):
    """Base class for every error raised by frame.validation."""


class UnsupportedSubsetError(ValidationError):
    """Raised when a benchmark subset outside the supported set is requested."""


class SampleIndexError(ValidationError):
    """Raised when a requested sample index is out of range for a subset."""


class ShapeMismatchError(ValidationError):
    """Raised when tensors/arrays handed to a validation step have incompatible shapes."""


class BenchmarkFormatError(ValidationError):
    """Raised when a downloaded benchmark subset's structure doesn't match what this
    adapter was built against (e.g. an unexpected band count) -- a defensive check
    against the upstream dataset format changing under us."""
