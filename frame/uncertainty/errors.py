"""Exception types raised by the FRAME test-time-augmentation uncertainty layer.

Same convention as `frame.preprocessing.errors`, `frame.geospatial.errors`,
`frame.consistency.errors`, and `frame.validation.errors`: a structurally
malformed configuration (an empty transform list, a transform that doesn't
round-trip, a shape it produces that doesn't match what the ensemble
expects) raises a specific, named error immediately.
"""


class UncertaintyError(Exception):
    """Base class for every error raised by frame.uncertainty."""


class InvalidTransformError(UncertaintyError):
    """Raised when a transform does not round-trip (forward -> inverse) correctly,
    or is otherwise malformed."""


class InvalidEnsembleConfigError(UncertaintyError):
    """Raised when the ensemble configuration itself is invalid (e.g. an empty
    transform list, N < 1)."""


class ShapeMismatchError(UncertaintyError):
    """Raised when tensors handed to a statistics/ensemble step have incompatible shapes."""
