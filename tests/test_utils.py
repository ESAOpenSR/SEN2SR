"""Tests for image tile iteration and edge alignment."""

import pytest
import torch

from sen2sr.utils import define_iteration, fix_lastchunk


@pytest.mark.parametrize("dimension", [(64, 64), (128, 256), (256, 128), (221, 161)])
def test_define_iteration_covers_image(dimension):
    coverage = torch.zeros(dimension, dtype=torch.bool)

    for y, x in define_iteration(dimension, chunk_size=128, overlap=32):
        assert 0 <= y <= max(dimension[0] - 128, 0)
        assert 0 <= x <= max(dimension[1] - 128, 0)
        coverage[y : y + 128, x : x + 128] = True

    assert coverage.all()


def test_fix_lastchunk_aligns_rectangular_image_edges():
    assert fix_lastchunk([(0, 0), (96, 192)], (221, 256), 128) == [(0, 0), (93, 128)]


@pytest.mark.parametrize("dimension", [(0, 128), (128, 0), (-1, 128)])
def test_define_iteration_rejects_empty_dimensions(dimension):
    with pytest.raises(ValueError, match="dimension"):
        define_iteration(dimension, chunk_size=128)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_define_iteration_rejects_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        define_iteration((128, 128), chunk_size=chunk_size)


@pytest.mark.parametrize("overlap", [-1, 128, 0.5])
def test_define_iteration_validates_overlap_even_for_small_images(overlap):
    with pytest.raises(ValueError, match="overlap"):
        define_iteration((64, 64), chunk_size=128, overlap=overlap)
