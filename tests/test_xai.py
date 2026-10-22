"""Basic attribution and saliency utility tests."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sen2sr.xai.lam import attr_grad, attribution_objective, create_lam_inputs, down_up
from sen2sr.xai.utils import gini, torch_gaussian_kde


def test_attribution_objective_passes_window_to_attribution_function():
    attribution = Mock(return_value=torch.tensor(7.0))
    image = torch.ones(1, 2, 4, 6)
    objective = attribution_objective(attribution, h=1, w=2, window=3)

    result = objective(image)

    assert result == 7
    attribution.assert_called_once_with(image, 1, 2, window=3)


@pytest.mark.parametrize("reduction,expected", [("sum", 120.0), ("mean", 5.0)])
def test_attr_grad_reduces_known_spatial_gradient(reduction, expected):
    rows, columns = torch.meshgrid(torch.arange(5), torch.arange(6), indexing="ij")
    image = (3 * rows + 4 * columns).float()[None, None].expand(2, 3, 5, 6)

    result = attr_grad(image, h=1, w=2, window=2, reduce=reduction)

    torch.testing.assert_close(result, torch.tensor(expected))


def test_attr_grad_rejects_unknown_reduction():
    with pytest.raises(ValueError, match="Invalid reduction type"):
        attr_grad(torch.ones(1, 1, 4, 4), h=0, w=0, window=2, reduce="median")


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_down_up_preserves_constant_image_and_shape(scale):
    image = torch.full((2, 3, 6, 9), 0.5)

    torch.testing.assert_close(down_up(image, scale_factor=scale), image)


def test_lam_inputs_drop_identity_scale_and_measure_blur_difference():
    image = torch.full((2, 6, 8), 0.5)

    cube, differences, scales = create_lam_inputs(image, ["1x", "2x", "3x"])

    assert scales == ["2x", "3x"]
    torch.testing.assert_close(cube, image[None].expand(2, 2, 6, 8))
    torch.testing.assert_close(differences, torch.zeros_like(cube))


@pytest.mark.parametrize(
    "values,expected", [([1.0, 1.0, 1.0], 0), ([0.0, 0.0, 0.0, 1.0], 0.75), ([1.0, 2.0, 3.0], 2 / 9)]
)
def test_gini_matches_known_distributions(values, expected):
    assert gini(values) == pytest.approx(expected, abs=1e-6)


def test_gini_is_invariant_to_order_and_does_not_mutate_input():
    values = np.array([3.0, -1.0, 2.0])
    original = values.copy()

    assert gini(values) == pytest.approx(gini(values[::-1]))
    np.testing.assert_array_equal(values, original)


def test_gaussian_kde_matches_single_weighted_point():
    point = torch.zeros(2, 1)
    grid = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    result = torch_gaussian_kde(point, weights=torch.tensor([2.0]), grid=grid, bandwidth=1)

    torch.testing.assert_close(result, torch.tensor([2.0, 2 * math.exp(-0.5)]))
