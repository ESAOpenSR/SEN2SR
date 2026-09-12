"""Validity (nodata / cloud / shadow) mask handling.

A masked pixel must never be confused with a genuine zero-reflectance
pixel: the reflectance tensor and the validity mask are always separate
objects that travel together through the pipeline.

The default invalid Scene Classification (SCL) codes below are ESA's own
published Sentinel-2 L2A SCL nomenclature (no data, saturated/defective,
cloud shadow, cloud medium probability, cloud high probability, thin
cirrus) -- this is a standard convention, not a threshold invented for this
project, and it is overridable by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_SCL_INVALID_CLASSES: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})


@dataclass(frozen=True)
class ValidityMask:
    """A boolean per-pixel mask: True = valid observation, False = masked."""

    array: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "array", np.asarray(self.array, dtype=bool))

    @classmethod
    def all_valid(cls, shape: tuple[int, int]) -> "ValidityMask":
        return cls(np.ones(shape, dtype=bool))

    @classmethod
    def from_nodata(cls, array: np.ndarray, nodata_value: float) -> "ValidityMask":
        """A pixel is invalid iff every band equals ``nodata_value`` there.

        ``array`` is a (bands, height, width) stack. A pixel that reads
        nodata in only *some* bands is still a real observation and stays
        valid.
        """
        is_nodata_per_band = array == nodata_value
        all_bands_nodata = np.all(is_nodata_per_band, axis=0)
        return cls(~all_bands_nodata)

    @classmethod
    def from_scl(cls, scl: np.ndarray, invalid_classes: Iterable[int] | None = None) -> "ValidityMask":
        """Build a mask from a Sentinel-2 L2A Scene Classification (SCL) band."""
        classes = DEFAULT_SCL_INVALID_CLASSES if invalid_classes is None else frozenset(invalid_classes)
        invalid = np.isin(scl, list(classes))
        return cls(~invalid)

    def combine(self, other: "ValidityMask") -> "ValidityMask":
        """Logical AND: valid only where both masks agree the pixel is valid."""
        return ValidityMask(self.array & other.array)

    def coverage(self) -> float:
        """Fraction of pixels that are valid (1.0 = fully valid, 0.0 = fully masked)."""
        return float(np.mean(self.array))
