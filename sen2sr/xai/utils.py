from typing import Union

import numpy as np
import torch


def torch_gaussian_kde(
    points: torch.Tensor, weights: torch.Tensor, grid: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    """
    Perform Kernel Density Estimation (KDE) using a Gaussian kernel with PyTorch.

    Args:
        points (torch.Tensor): A 2D tensor of shape (2, n_points) with [x, y] positions.
        weights (torch.Tensor): A 1D tensor of shape (n_points,) with weights for each point.
        grid (torch.Tensor): A 2D tensor of shape (2, n_grid_points) with [x, y] positions for the evaluation grid.
        bandwidth (float): The bandwidth (standard deviation) for the Gaussian kernel.

    Returns:
        torch.Tensor: A tensor representing KDE values evaluated at the grid positions.
    """
    # Compute pairwise squared distances between grid and data points
    distances = torch.cdist(grid.T, points.T, p=2) ** 2

    # Apply Gaussian kernel
    kernel_values = torch.exp(-distances / (2 * bandwidth**2))

    # Weight and sum the kernel values to get the KDE
    kde_values = (kernel_values * weights).sum(dim=1)

    return kde_values


def vis_saliency_kde(
    map: torch.Tensor, scale: int = 4, bandwidth: float = 1.0
) -> torch.Tensor:
    """
    Visualize saliency map KDE using a Gaussian kernel.

    Args:
        map (torch.Tensor): A 2D tensor representing the saliency map.
        scale (int): Scaling factor for the output density map.
        bandwidth (float): Bandwidth for the KDE Gaussian kernel.

    Returns:
        torch.Tensor: A normalized 2D tensor representing the KDE of the saliency map.
    """
    # Flatten the saliency map and prepare coordinates
    grad_flat = map.flatten()
    datapoint_y, datapoint_x = torch.meshgrid(
        torch.arange(map.shape[0], dtype=torch.float32),
        torch.arange(map.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.vstack([datapoint_x.flatten(), datapoint_y.flatten()])

    # Generate grid for KDE evaluation
    Y, X = torch.meshgrid(
        torch.arange(0, map.shape[0], dtype=torch.float32),
        torch.arange(0, map.shape[1], dtype=torch.float32),
        indexing="ij",
    )
    grid_positions = torch.vstack([X.flatten(), Y.flatten()])

    # Perform KDE on the grid
    kde_values = torch_gaussian_kde(
        pixels, grad_flat, grid_positions, bandwidth=bandwidth
    )

    # Reshape and normalize KDE output
    Z = kde_values.reshape(map.shape)
    maximum = Z.max()
    if maximum > 0:
        Z = Z / maximum

    # Reshape to the SR scale
    Z = torch.nn.functional.interpolate(
        Z[None, None], scale_factor=scale, mode="bicubic", antialias=True
    ).squeeze()

    return Z


def gini(array: Union[np.ndarray, list]) -> float:
    """
    Calculate the Gini coefficient of a 1-dimensional array. The Gini coefficient is a measure of inequality
    where 0 represents perfect equality and 1 represents maximal inequality.

    Args:
        array (Union[np.ndarray, list]): A 1D array or list of numerical values for which the Gini coefficient is calculated.

    Returns:
        float: The Gini coefficient, a value between 0 and 1.

    Notes:
        - This implementation is based on the formula for the Gini coefficient described here:
          http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        - The input array is treated as a 1-dimensional array.
        - All values in the array must be non-negative. Negative values are shifted to zero if present.
        - Zero values are adjusted slightly to avoid division by zero.

    """
    # Ensure array is a flattened 1D numpy array
    array = np.asarray(array, dtype=np.float64).flatten()

    # Shift values if there are any negative elements, as Gini requires non-negative values
    if np.amin(array) < 0:
        array -= np.amin(array)

    # Avoid division by zero by slightly adjusting zero values
    array += 1e-7

    # Sort array values in ascending order for the Gini calculation
    array = np.sort(array)

    # Create an index array (1-based) for each element in the sorted array
    index = np.arange(1, array.shape[0] + 1)

    # Calculate the number of elements in the array
    n = array.shape[0]

    # Compute the Gini coefficient using the sorted values and index-based formula
    gini_coefficient = (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    return gini_coefficient
