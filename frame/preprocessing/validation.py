"""Input validation for the FRAME preprocessing layer.

These functions check structural properties of an input (shape, band set,
resolution) and raise a specific, named error the moment something doesn't
match what the downstream SR path requires. Nothing here silently coerces
or guesses -- an unsupported input is rejected, not "fixed."
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from frame.preprocessing.errors import InvalidShapeError, UnsupportedBandsError, UnsupportedResolutionError


def validate_shape(array: np.ndarray, expected_size: int | None = None) -> None:
    """Validate that ``array`` is a (bands, height, width) stack.

    Args:
        array: Candidate input array.
        expected_size: If given, height and width must both equal this
            value exactly (the model's proven native patch size). If
            ``None``, only a square, non-empty (C, H, W) shape is required.
    """
    if array.ndim != 3:
        raise InvalidShapeError(
            f"Expected a 3-D (bands, height, width) array, got {array.ndim}-D "
            f"with shape {array.shape}."
        )

    _, height, width = array.shape
    if height == 0 or width == 0 or array.shape[0] == 0:
        raise InvalidShapeError(f"Array has a zero-sized dimension: shape {array.shape}.")

    if height != width:
        raise InvalidShapeError(
            f"Expected a square patch (height == width), got height={height}, width={width}."
        )

    if expected_size is not None and height != expected_size:
        raise InvalidShapeError(
            f"Expected a {expected_size}x{expected_size} patch, got {height}x{width}."
        )


def validate_bands(band_names: Sequence[str], expected_bands: Sequence[str]) -> None:
    """Validate that ``band_names`` is exactly ``expected_bands`` as a set (any order).

    Rejects missing bands, unexpected extra bands, and duplicate bands --
    each with a message naming the offending band(s).
    """
    if len(band_names) != len(set(band_names)):
        duplicates = sorted({b for b in band_names if band_names.count(b) > 1})
        raise UnsupportedBandsError(f"Duplicate band(s) in input: {duplicates}.")

    actual = set(band_names)
    expected = set(expected_bands)

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)

    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing required band(s) {missing}")
        if unexpected:
            parts.append(f"unexpected band(s) {unexpected}")
        raise UnsupportedBandsError(
            f"Band set does not match the required {sorted(expected)}: {'; '.join(parts)}."
        )


def reorder_bands(array: np.ndarray, band_names: Sequence[str], expected_order: Sequence[str]) -> np.ndarray:
    """Reindex ``array`` along its band axis (axis 0) to match ``expected_order``.

    Raises ``UnsupportedBandsError`` if any band in ``expected_order`` is not
    present in ``band_names``.
    """
    index_of = {name: i for i, name in enumerate(band_names)}
    missing = [b for b in expected_order if b not in index_of]
    if missing:
        raise UnsupportedBandsError(
            f"Cannot reorder to {list(expected_order)}: missing band(s) {missing} in input {list(band_names)}."
        )
    indices = [index_of[b] for b in expected_order]
    return array[indices, ...]


def validate_resolution(resolution_m: float, expected_resolution_m: float, tolerance: float = 1e-6) -> None:
    """Validate that ``resolution_m`` matches ``expected_resolution_m`` within a
    floating-point tolerance (not a scientific tolerance -- just epsilon for
    float comparison)."""
    if abs(resolution_m - expected_resolution_m) > tolerance:
        raise UnsupportedResolutionError(
            f"Expected {expected_resolution_m} m resolution, got {resolution_m} m."
        )
