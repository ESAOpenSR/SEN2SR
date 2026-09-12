"""Tests for frame.uncertainty.transforms -- the geometric test-time
augmentation transforms used by stochastic TTA uncertainty (Phase 5).

Every transform must round-trip exactly (forward then inverse == identity),
preserve tensor shape after the round trip, work on (C, H, W) tensors, and
carry no hidden random state.
"""

import torch
import pytest

from frame.uncertainty.errors import InvalidTransformError
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


def _chw(seed=0, c=4, h=8, w=8):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(c, h, w, generator=g)


# ---------------------------------------------------------------------------
# Individual transforms: forward -> inverse == identity
# ---------------------------------------------------------------------------

def test_identity_transform_is_exact():
    x = _chw()
    assert torch.equal(IDENTITY.forward(x), x)
    assert torch.equal(IDENTITY.inverse(x), x)


def test_horizontal_flip_round_trips_exactly():
    x = _chw()
    flipped = HFLIP.forward(x)
    assert not torch.equal(flipped, x)  # actually changed something
    restored = HFLIP.inverse(flipped)
    assert torch.equal(restored, x)


def test_vertical_flip_round_trips_exactly():
    x = _chw()
    flipped = VFLIP.forward(x)
    assert not torch.equal(flipped, x)
    restored = VFLIP.inverse(flipped)
    assert torch.equal(restored, x)


def test_rot90_round_trips_exactly():
    x = _chw()
    rotated = ROT90.forward(x)
    restored = ROT90.inverse(rotated)
    assert torch.equal(restored, x)


def test_rot180_round_trips_exactly():
    x = _chw()
    rotated = ROT180.forward(x)
    restored = ROT180.inverse(rotated)
    assert torch.equal(restored, x)


def test_rot270_round_trips_exactly():
    x = _chw()
    rotated = ROT270.forward(x)
    restored = ROT270.inverse(rotated)
    assert torch.equal(restored, x)


def test_rot90_and_rot270_are_mutual_inverses_in_effect():
    x = _chw()
    # rotating 90 then 270 (or vice versa) should return to the original
    assert torch.equal(ROT270.forward(ROT90.forward(x)), x)
    assert torch.equal(ROT90.forward(ROT270.forward(x)), x)


def test_rot90_changes_spatial_shape_for_non_square_input_but_still_round_trips():
    x = _chw(h=8, w=12)
    rotated = ROT90.forward(x)
    assert rotated.shape == (4, 12, 8)  # H/W swapped
    restored = ROT90.inverse(rotated)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)


# ---------------------------------------------------------------------------
# Shape / band-order preservation on the (forward -> inverse) round trip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("transform", [IDENTITY, HFLIP, VFLIP, ROT90, ROT180, ROT270])
def test_round_trip_preserves_shape_for_square_input(transform):
    x = _chw(h=8, w=8)
    assert transform.inverse(transform.forward(x)).shape == x.shape


@pytest.mark.parametrize("transform", [IDENTITY, HFLIP, VFLIP, ROT90, ROT180, ROT270])
def test_round_trip_preserves_band_order(transform):
    # give each band a distinct, checkable constant value
    x = torch.stack([torch.full((8, 8), float(b)) for b in range(4)])
    restored = transform.inverse(transform.forward(x))
    for b in range(4):
        assert torch.all(restored[b] == float(b))


def test_transforms_operate_on_3d_chw_tensors_only():
    x = torch.rand(8, 8)  # missing band axis
    with pytest.raises(InvalidTransformError):
        HFLIP.forward(x)


# ---------------------------------------------------------------------------
# No hidden random state
# ---------------------------------------------------------------------------

def test_transforms_are_deterministic_across_repeated_calls():
    x = _chw()
    for t in DEFAULT_TRANSFORMS:
        a = t.forward(x)
        b = t.forward(x)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# Registry / recording
# ---------------------------------------------------------------------------

def test_default_transforms_includes_identity():
    assert IDENTITY in DEFAULT_TRANSFORMS


def test_default_transforms_has_the_six_documented_geometric_transforms():
    names = {t.name for t in DEFAULT_TRANSFORMS}
    assert names == {"identity", "hflip", "vflip", "rot90", "rot180", "rot270"}


def test_each_transform_has_a_recorded_name():
    for t in DEFAULT_TRANSFORMS:
        assert isinstance(t.name, str) and len(t.name) > 0


# ---------------------------------------------------------------------------
# validate_round_trip -- used by the ensemble to reject a misconfigured transform
# ---------------------------------------------------------------------------

def test_validate_round_trip_accepts_every_default_transform():
    x = _chw()
    for t in DEFAULT_TRANSFORMS:
        validate_round_trip(t, x)  # must not raise


def test_validate_round_trip_rejects_a_broken_transform():
    broken = Transform(name="broken", forward=lambda x: x.flip(-1), inverse=lambda x: x)  # not a true inverse
    x = _chw()
    with pytest.raises(InvalidTransformError):
        validate_round_trip(broken, x)
