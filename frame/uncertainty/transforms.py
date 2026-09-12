"""Geometric test-time-augmentation transforms for stochastic uncertainty
(docs/FRAME_TECHNICAL_SPEC.md Section 13's recommended first uncertainty
method: test-time perturbation ensembling).

Each `Transform` is a pure, deterministic pair of functions on (C, H, W)
tensors: `forward` (applied to the LR input before inference) and `inverse`
(applied to the model's SR output afterward, so every ensemble member's
prediction lands back in the SAME canonical orientation before being
compared/averaged). Every transform here is a lossless geometric symmetry
(a flip or a multiple-of-90-degree rotation) -- it changes *which pixel
sits where*, never a pixel's own reflectance value, so it preserves the
scene's semantic content exactly, only testing whether the frozen model's
prediction is *stable* under a re-framing of the same real observation.

No transform here carries hidden random state -- `forward`/`inverse` are
plain, seed-free functions. (An optional, off-by-default noise-based
perturbation lives in a separate, clearly-labeled place -- see
frame/uncertainty/README.md's "Optional noise perturbation" section; it is
NOT part of `DEFAULT_TRANSFORMS`.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from frame.uncertainty.errors import InvalidTransformError


@dataclass(frozen=True)
class Transform:
    name: str
    forward: Callable[[torch.Tensor], torch.Tensor]
    inverse: Callable[[torch.Tensor], torch.Tensor]


def _require_chw(x: torch.Tensor, op_name: str) -> None:
    if x.ndim != 3:
        raise InvalidTransformError(
            f"Transform op {op_name!r} requires a 3-D (bands, H, W) tensor, got {x.ndim}-D shape {tuple(x.shape)}."
        )


def _identity(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "identity")
    return x


def _hflip(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "hflip")
    return torch.flip(x, dims=(-1,))


def _vflip(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "vflip")
    return torch.flip(x, dims=(-2,))


def _rot90(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "rot90")
    return torch.rot90(x, k=1, dims=(-2, -1))


def _rot90_inv(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "rot90_inverse")
    return torch.rot90(x, k=-1, dims=(-2, -1))


def _rot180(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "rot180")
    return torch.rot90(x, k=2, dims=(-2, -1))


def _rot270(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "rot270")
    return torch.rot90(x, k=-1, dims=(-2, -1))


def _rot270_inv(x: torch.Tensor) -> torch.Tensor:
    _require_chw(x, "rot270_inverse")
    return torch.rot90(x, k=1, dims=(-2, -1))


# The identity transform must always be included (per Phase 5's explicit
# requirement) -- it is both a valid ensemble member and the reference
# orientation every other transform's inverse restores to.
IDENTITY = Transform("identity", _identity, _identity)

# Flips are self-inverse (flipping twice returns the original).
HFLIP = Transform("hflip", _hflip, _hflip)
VFLIP = Transform("vflip", _vflip, _vflip)

# Rotations: rot90's inverse is rot270 (i.e. k=-1); rot180 is self-inverse
# (180 + 180 = 360); rot270's inverse is rot90 (k=+1).
ROT90 = Transform("rot90", _rot90, _rot90_inv)
ROT180 = Transform("rot180", _rot180, _rot180)
ROT270 = Transform("rot270", _rot270, _rot270_inv)

# The 6 lossless geometric symmetries used by default -- identity, both
# axis flips, and the three non-trivial multiples of 90 degrees. This is a
# deliberate subset of the full 8-element dihedral group D4 (it omits the
# two diagonal transpose-flips) -- exactly the set named in the Phase 5
# task description, not the full symmetry group.
DEFAULT_TRANSFORMS: Tuple[Transform, ...] = (IDENTITY, HFLIP, VFLIP, ROT90, ROT180, ROT270)


def validate_round_trip(transform: Transform, sample: torch.Tensor) -> None:
    """Raise InvalidTransformError unless `transform.inverse(transform.forward(sample))`
    exactly reproduces `sample` -- a cheap sanity check run once per transform
    before committing GPU time to a full ensemble pass with it."""
    try:
        forwarded = transform.forward(sample)
        restored = transform.inverse(forwarded)
    except InvalidTransformError:
        raise
    except Exception as exc:  # noqa: BLE001 -- wrapped with our own taxonomy, original preserved
        raise InvalidTransformError(f"Transform {transform.name!r} raised while round-tripping: {exc}") from exc

    if restored.shape != sample.shape:
        raise InvalidTransformError(
            f"Transform {transform.name!r} does not round-trip shape: {tuple(sample.shape)} -> "
            f"{tuple(forwarded.shape)} -> {tuple(restored.shape)}."
        )
    if not torch.equal(restored, sample):
        raise InvalidTransformError(f"Transform {transform.name!r} does not exactly round-trip (forward -> inverse != identity).")
