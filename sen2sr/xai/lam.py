from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

from sen2sr.xai.utils import gini, vis_saliency_kde


def attribution_objective(attr_func, h: int, w: int, window: int = 16):
    """
    Creates an objective function to calculate attribution within a specified window
    at given coordinates using an attribution function.

    Args:
        attr_func (Callable): A function that calculates attributions for an image.
        h (int): The top coordinate of the window within the image.
        w (int): The left coordinate of the window within the image.
        window (int, optional): The size of the square window. Defaults to 16.

    Returns:
        Callable: A function that takes an image as input and computes the attribution
        at the specified window location.
    """

    def calculate_objective(image):
        """
        Computes the attribution for a specified window within the given image.

        Args:
            image (torch.Tensor): A tensor representing the input image.

        Returns:
            torch.Tensor: The calculated attribution value within the specified window.
        """
        return attr_func(image, h, w, window=window)

    return calculate_objective


def attr_grad(
    tensor: torch.Tensor,
    h: int,
    w: int,
    window: int = 8,
    reduce: str = "sum",
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Computes the gradient magnitude within a specified window of a 4D tensor and reduces the result.

    Args:
        tensor (torch.Tensor): A 4D tensor of shape (batch_size, channels, height, width).
        h (int): Starting height position of the window within the tensor.
        w (int): Starting width position of the window within the tensor.
        window (int, optional): The size of the square window to extract. Defaults to 8.
        reduce (str, optional): The reduction operation to apply to the window ('sum' or 'mean'). Defaults to 'sum'.
        scale (float, optional): Scaling factor to apply to the gradient magnitude. Defaults to 1.0.

    Returns:
        torch.Tensor: The reduced gradient magnitude for the specified window.
    """

    # Get tensor dimensions
    height = tensor.size(2)
    width = tensor.size(3)

    # Compute horizontal gradients by taking the difference between adjacent rows
    h_grad = torch.pow(tensor[:, :, : height - 1, :] - tensor[:, :, 1:, :], 2)

    # Compute vertical gradients by taking the difference between adjacent columns
    w_grad = torch.pow(tensor[:, :, :, : width - 1] - tensor[:, :, :, 1:], 2)

    # Calculate gradient magnitude by summing squares of gradients and taking the square root
    grad_magnitude = torch.sqrt(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :])

    # Crop the gradient magnitude tensor to the specified window
    windowed_grad = grad_magnitude[:, :, h : h + window, w : w + window]

    # Apply reduction (sum or mean) to the cropped window
    if reduce == "sum":
        return torch.sum(windowed_grad)
    elif reduce == "mean":
        return torch.mean(windowed_grad)
    else:
        raise ValueError(f"Invalid reduction type: {reduce}. Use 'sum' or 'mean'.")


def down_up(X: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
    """Downsample and upsample an image using bilinear interpolation.

    Args:
        X (torch.Tensor): The input tensor (Bands x Height x Width).
        scale_factor (float, optional): The scaling factor. Defaults to 0.5.

    Returns:
        torch.Tensor: The downsampled and upsampled image.
    """
    shape_init = X.shape
    return torch.nn.functional.interpolate(
        input=torch.nn.functional.interpolate(
            input=X, scale_factor=1 / scale_factor, mode="bilinear", antialias=True
        ),
        size=shape_init[2:],
        mode="bilinear",
        antialias=True,
    )


def create_blur_cube(X: torch.Tensor, scales: list) -> torch.Tensor:
    """Create a cube of blurred images at different scales.

    Args:
        X (torch.Tensor): The input tensor (Bands x Height x Width).
        scales (list): The scales to evaluate.

    Returns:
        torch.Tensor: The cube of blurred images.
    """
    scales_int = [float(scale[:-1]) for scale in scales]
    return torch.stack([down_up(X[None], scale) for scale in scales_int]).squeeze(1)


def create_lam_inputs(
    X: torch.Tensor, scales: list
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Create the inputs for the Local Attribution Map (LAM).

    Args:
        X (torch.Tensor): The input tensor (Bands x Height x Width).
        scales (list): The scales to evaluate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, list]: The cube of blurred
            images, the difference between the input and the cube,
            and the scales.
    """
    cube = create_blur_cube(X, scales)
    diff = torch.abs(X[None] - cube)
    return cube[1:], diff[1:], scales[1:]


def lam(
    X: torch.Tensor,
    model: torch.nn.Module,    
    h: int = 240,
    w: int = 240,
    window: int = 32,
    scales: list = ["1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x"],
) -> Tuple[np.ndarray, float, float, np.ndarray]:
    """Estimate the Local Attribution Map (LAM)

    Args:
        X (torch.Tensor): The input tensor (Bands x Height x Width).
        model (torch.nn.Module): The model to evaluate.
        model_scale (float, optional): The scale of the model. Defaults to 4.
        h (int, optional): The height of the window to evaluate. Defaults to 240.
        w (int, optional): The width of the window to evaluate. Defaults to 240.
        window (int, optional): The window size. Defaults to 32.
        scales (list, optional): The scales to evaluate. Defaults to
            ["1x", "2x", "3x", "4x", "5x", "6x", "7x", "8x"].

    Returns:
        Tuple[np.ndarray, float, float, np.ndarray]: _description_
    """

    # Create the LAM inputs
    cube, diff, scales = create_lam_inputs(X, scales)

    # Create the attribution objective function
    attr_objective = attribution_objective(attr_grad, h, w, window=window)

    # Initialize the gradient accumulation list
    grad_accumulate_list = torch.zeros_like(cube).cpu().numpy()

    # Compute gradient for each interpolated image
    for i in tqdm(range(cube.shape[0]), desc="Computing gradients"):
        
        # Convert interpolated image to tensor and set requires_grad for backpropagation
        img_tensor = cube[i].float()[None]
        img_tensor.requires_grad_(True)

        # Forward pass through the model and compute attribution objective
        result = model(img_tensor)
        target = attr_objective(result)
        target.backward()  # Compute gradients

        # determine the scale of the model
        if i == 0:
            scale_factor = result.shape[2] / img_tensor.shape[2]

        # Extract gradient, handling NaNs if present
        grad = img_tensor.grad.cpu().numpy()
        grad = np.nan_to_num(grad)  # Replace NaNs with 0

        # Accumulate gradients adjusted by lambda derivatives
        grad_accumulate_list[i] = grad * diff[i].cpu().numpy()

    # Sum the accumulated gradients across all bands
    lam_results = torch.sum(torch.from_numpy(np.abs(grad_accumulate_list)), dim=0)
    grad_2d = lam_results.sum(dim=0).abs().numpy()
    grad_max = grad_2d.max()
    grad_norm = grad_2d / grad_max if grad_max > 0 else np.zeros_like(grad_2d)

    # Estimate gini index
    gini_index = gini(grad_norm.flatten())

    ## window to image size
    # ratio_img_to_window = (X.shape[1] * model_scale) // window

    # KDE estimation
    density = vis_saliency_kde(torch.from_numpy(grad_norm), scale=scale_factor, bandwidth=1.0)
    kde_map = np.log1p(density.cpu().numpy())
    complexity_metric = (1 - gini_index) * 100  # / ratio_img_to_window

    # Estimate blurriness sensitivity
    robustness_vector = np.abs(grad_accumulate_list).mean(axis=(1, 2, 3))
    robustness_metric = np.trapezoid(robustness_vector)

    # Return the LAM results
    return kde_map, complexity_metric, robustness_metric, robustness_vector
