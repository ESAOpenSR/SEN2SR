"""Basic attribution and saliency utility tests."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from sen2sr.xai.lam import attr_grad, attribution_objective, create_blur_cube, create_lam_inputs, down_up, lam
from sen2sr.xai.utils import gini, torch_gaussian_kde, vis_saliency_kde


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


@pytest.mark.parametrize("values", [[1, 2, 3], np.array([1, 2, 3]), [0, 0, 0]])
def test_gini_accepts_integer_inputs(values):
    expected = 0 if np.max(values) == 0 else 2 / 9

    assert gini(values) == pytest.approx(expected, abs=1e-6)


def test_gaussian_kde_matches_single_weighted_point():
    point = torch.zeros(2, 1)
    grid = torch.tensor([[0.0, 1.0], [0.0, 0.0]])

    result = torch_gaussian_kde(point, weights=torch.tensor([2.0]), grid=grid, bandwidth=1)

    torch.testing.assert_close(result, torch.tensor([2.0, 2 * math.exp(-0.5)]))


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("scales", [["1x"], ["1x", "2x", "3x"]])
def test_blur_cube_preserves_scale_and_channel_axes(channels, scales):
    image = torch.full((channels, 6, 8), 0.5)

    result = create_blur_cube(image, scales)

    torch.testing.assert_close(result, image[None].expand(len(scales), channels, 6, 8))


def test_zero_saliency_map_produces_finite_zero_density():
    result = vis_saliency_kde(torch.zeros(3, 5), scale=2)

    torch.testing.assert_close(result, torch.zeros(6, 10), rtol=0, atol=0)


def test_saliency_density_is_invariant_to_positive_weight_scale():
    saliency = torch.zeros(5, 7)
    saliency[2, 3] = 1

    torch.testing.assert_close(vis_saliency_kde(saliency), vis_saliency_kde(7 * saliency))


@pytest.mark.parametrize("channels", [1, 2])
def test_lam_end_to_end_produces_finite_maps_and_metrics(channels):
    image = torch.arange(channels * 6 * 8, dtype=torch.float32).reshape(channels, 6, 8) / 100
    original = image.clone()
    model = torch.nn.Upsample(scale_factor=2, mode="nearest")

    density, complexity, robustness, per_scale = lam(image, model, h=2, w=2, window=2, scales=["1x", "2x", "3x"])

    assert isinstance(density, np.ndarray)
    assert density.shape == (12, 16)
    assert np.isfinite(density).all()
    assert np.isfinite(complexity)
    assert 0 <= complexity <= 100
    assert np.isfinite(robustness)
    assert robustness >= 0
    assert per_scale.shape == (2,)
    assert np.isfinite(per_scale).all()
    assert (per_scale >= 0).all()
    torch.testing.assert_close(image, original)


def test_constant_image_has_zero_lam_and_robustness():
    model = torch.nn.Upsample(scale_factor=2, mode="nearest")

    density, complexity, robustness, per_scale = lam(
        torch.ones(2, 6, 8), model, h=2, w=2, window=2, scales=["1x", "2x", "3x"]
    )

    np.testing.assert_array_equal(density, np.zeros((12, 16)))
    np.testing.assert_array_equal(per_scale, np.zeros(2))
    assert np.isfinite(complexity)
    assert robustness == 0
