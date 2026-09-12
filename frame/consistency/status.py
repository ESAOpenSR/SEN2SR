"""The raw diagnostic status used throughout frame.consistency.

Per docs/FRAME_TECHNICAL_SPEC.md Section 9.2/11: this project does not invent
numeric pass/fail thresholds for spectral-consistency diagnostics -- that
would require an empirical discrepancy distribution from real runs this
project does not have yet. `ComputationStatus` is therefore NOT a verdict
("good"/"bad") -- it only states whether a number could be computed at all:

- COMPUTABLE     -- enough valid data existed; the reported statistics are
                     real numbers.
- NOT_COMPUTABLE -- the inputs were well-formed, but there was nothing (or
                     nothing valid, e.g. every relevant pixel was masked, or
                     two tiles had no geometric overlap) to compute a
                     statistic from. Reported statistics are None, never a
                     fabricated 0.0 or a bare NaN.
- INVALID_INPUT  -- the inputs, while not so malformed that the call raised
                     an exception, describe a scenario this diagnostic
                     cannot meaningfully evaluate (e.g. two "overlapping"
                     tiles whose declared offsets do not actually overlap in
                     pixel space, or mismatched band counts between two
                     tiles). Used by frame.consistency.tiles specifically --
                     see its module docstring for why a status, rather than
                     a raised exception, is preferable there.

Never converted into a boolean "pass" anywhere in this package.
"""

from __future__ import annotations

from enum import Enum


class ComputationStatus(str, Enum):
    COMPUTABLE = "COMPUTABLE"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"
    INVALID_INPUT = "INVALID_INPUT"
