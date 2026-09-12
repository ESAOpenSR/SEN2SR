"""FRAME stochastic test-time-augmentation uncertainty layer.

Runs FRAME's proven, **unmodified** `SEN2SRLite/NonReference_RGBN_x4` model
N times on geometrically-transformed views of the SAME input, undoes each
transform on the corresponding SR output, and reports the per-pixel
dispersion across the (de-transformed, therefore pixel-aligned) predictions
as a **relative, architecture-conditioned model-stability uncertainty
proxy** -- docs/FRAME_TECHNICAL_SPEC.md Section 13's recommended first
uncertainty method (stochastic inference / test-time perturbation
ensembling), now implemented.

This is emphatically NOT a calibrated probability of error, a confidence
interval, or a physically rigorous uncertainty bound -- see README.md for
the full, repeated caveat -- and it is NOT LAM
(`sen2sr/xai/lam.py`, explainability/sensitivity only, a different
question answered by a different mechanism -- see README.md's "TTA vs.
LAM" section).

This package imports nothing from `sen2sr` and calls no model itself -- the
real model call happens in `experiments/uncertainty/run_experiment.py`,
matching the separation of concerns established in Phases 1-4.

Public API:
    run_stochastic_uncertainty, UncertaintyResult, TransformDisagreement,
        SCALAR_SUMMARY_DEFINITION                          (report.py)
    run_tta_ensemble, EnsembleRunResult                      (ensemble.py)
    Transform, DEFAULT_TRANSFORMS, IDENTITY, HFLIP, VFLIP,
        ROT90, ROT180, ROT270, validate_round_trip            (transforms.py)
    WelfordAccumulator, UncertaintyDistributionStats,
        compute_distribution_stats, normalize_for_visualization (statistics.py)
    Exceptions: UncertaintyError and its subclasses (see .errors)
"""

from frame.uncertainty.ensemble import EnsembleRunResult, run_tta_ensemble
from frame.uncertainty.errors import (
    InvalidEnsembleConfigError,
    InvalidTransformError,
    ShapeMismatchError,
    UncertaintyError,
)
from frame.uncertainty.report import (
    SCALAR_SUMMARY_DEFINITION,
    TransformDisagreement,
    UncertaintyResult,
    run_stochastic_uncertainty,
)
from frame.uncertainty.statistics import (
    UncertaintyDistributionStats,
    WelfordAccumulator,
    compute_distribution_stats,
    normalize_for_visualization,
)
from frame.uncertainty.transforms import (
    DEFAULT_TRANSFORMS,
    HFLIP,
    IDENTITY,
    ROT90,
    ROT180,
    ROT270,
    VFLIP,
    Transform,
    validate_round_trip,
)

__all__ = [
    "run_stochastic_uncertainty",
    "UncertaintyResult",
    "TransformDisagreement",
    "SCALAR_SUMMARY_DEFINITION",
    "run_tta_ensemble",
    "EnsembleRunResult",
    "Transform",
    "DEFAULT_TRANSFORMS",
    "IDENTITY",
    "HFLIP",
    "VFLIP",
    "ROT90",
    "ROT180",
    "ROT270",
    "validate_round_trip",
    "WelfordAccumulator",
    "UncertaintyDistributionStats",
    "compute_distribution_stats",
    "normalize_for_visualization",
    "UncertaintyError",
    "InvalidTransformError",
    "InvalidEnsembleConfigError",
    "ShapeMismatchError",
]
