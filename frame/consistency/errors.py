"""Exception types raised by the FRAME spectral-consistency diagnostics layer.

Structural problems with the inputs handed to a diagnostic (wrong shape, a
scale factor that does not actually relate the two tensors, a mask that does
not match) are rejected loudly and specifically, the same convention used by
`frame.preprocessing.errors` and `frame.geospatial.errors`. This is
deliberately different from `ComputationStatus.NOT_COMPUTABLE` /
`ComputationStatus.INVALID_INPUT` (see `frame.consistency.status`), which
describe a *well-formed* diagnostic call that simply had nothing (or nothing
valid) to measure -- not a malformed call.
"""


class ConsistencyError(Exception):
    """Base class for every error raised by frame.consistency."""


class ShapeMismatchError(ConsistencyError):
    """Raised when tensors/arrays handed to a diagnostic have incompatible shapes."""


class ScaleFactorError(ConsistencyError):
    """Raised when a declared scale factor does not relate the SR and LR shapes."""


class InvalidMaskError(ConsistencyError):
    """Raised when a validity mask's shape or dtype does not match what it must cover."""


class MissingBandError(ConsistencyError):
    """Raised when a band name required for a spectral-index comparison is absent."""
