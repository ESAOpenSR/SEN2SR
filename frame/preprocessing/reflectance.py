"""Reflectance normalization for the FRAME preprocessing layer.

Reproduces exactly the scaling used by the upstream sen2sr README examples
and by ``experiments/baseline/run_baseline.py``:

    X = (digital_number / 10_000).astype("float32")
    X = nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

Whether an input still needs the ``/10000`` scaling is never guessed from
the data itself (e.g. via a magnitude heuristic) -- the caller must state
it explicitly via ``input_scale``, so a caller can never silently apply the
scaling twice (or not at all) by accident.
"""

from __future__ import annotations

import numpy as np

from frame.preprocessing.errors import InvalidInputScaleError

RAW_DIGITAL_NUMBER = "raw_digital_number"
REFLECTANCE = "reflectance"
_VALID_INPUT_SCALES = (RAW_DIGITAL_NUMBER, REFLECTANCE)

SENTINEL2_L2A_REFLECTANCE_SCALE = 10_000.0


def to_reflectance(array: np.ndarray, input_scale: str) -> np.ndarray:
    """Convert ``array`` to float32 surface-reflectance-like values in [0, 1].

    Args:
        array: Input band stack, any numeric dtype.
        input_scale: Either ``"raw_digital_number"`` (Sentinel-2 L2A integer
            digital numbers, scaled by 10,000 as documented by the upstream
            package) or ``"reflectance"`` (already float reflectance in
            [0, 1] -- left numerically unscaled). No other value is
            accepted.

    Returns:
        A new float32 array, same shape as ``array``, with NaN/+Inf/-Inf
        replaced by 0.0.
    """
    if input_scale not in _VALID_INPUT_SCALES:
        raise InvalidInputScaleError(
            f"Unknown input_scale {input_scale!r}; expected one of {_VALID_INPUT_SCALES}."
        )

    array = np.asarray(array)

    if input_scale == RAW_DIGITAL_NUMBER:
        # Match experiments/baseline/run_baseline.py exactly: divide in the
        # input's own precision (numpy promotes to float64), THEN cast to
        # float32 -- not the other way around, to avoid introducing any
        # avoidable discrepancy against the proven Baseline 0 numerics.
        result = (array / SENTINEL2_L2A_REFLECTANCE_SCALE).astype("float32")
    else:
        result = array.astype("float32")

    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")
