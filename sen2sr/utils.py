import itertools

import torch
import torch.nn.functional as F
from tqdm import tqdm


def define_iteration(dimension: tuple, chunk_size: int, overlap: int = 0):
    """
    Define the iteration strategy to walk through the image with an overlap.

    Args:
        dimension (tuple): Dimension of the S2 image.
        chunk_size (int): Size of the chunks.
        overlap (int): Size of the overlap between chunks.

    Returns:
        list: List of chunk coordinates.
    """
    dimy, dimx = dimension

    if dimy <= 0 or dimx <= 0:
        message = "dimension must contain positive height and width"
        raise ValueError(message)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        message = "chunk_size must be a positive integer"
        raise ValueError(message)
    if not isinstance(overlap, int) or not 0 <= overlap < chunk_size:
        message = "overlap must be an integer between 0 and chunk_size - 1"
        raise ValueError(message)

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    # Adjust step to create overlap
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, y_step), range(0, dimx, x_step)))

    # Fix chunks at the edges to stay within bounds
    iterchunks_fixed = fix_lastchunk(iterchunks=iterchunks, s2dim=dimension, chunk_size=chunk_size)

    return iterchunks_fixed


def fix_lastchunk(iterchunks, s2dim, chunk_size):
    """
    Fix the last chunk of the overlay to ensure it aligns with image boundaries.

    Args:
        iterchunks (list): List of chunks created by itertools.product.
        s2dim (tuple): Dimension of the S2 images.
        chunk_size (int): Size of the chunks.

    Returns:
        list: List of adjusted chunk coordinates.
    """
    itercontainer = []

    for index_i, index_j in iterchunks:
        # Adjust if the chunk extends beyond bounds
        if index_i + chunk_size > s2dim[0]:
            index_i = max(s2dim[0] - chunk_size, 0)
        if index_j + chunk_size > s2dim[1]:
            index_j = max(s2dim[1] - chunk_size, 0)

        itercontainer.append((index_i, index_j))

    return itercontainer


def predict_large(
    X: torch.Tensor,
    model: torch.nn.Module,
    overlap: int = 32,
) -> torch.Tensor:
    """Predict a (channels, height, width) image in overlapping 128-pixel tiles.

    The model must return (1, output_channels, 128 * scale, 128 * scale)
    with a consistent positive integer scale. Tiles smaller than 128 pixels
    are padded by repeating their edge values, then cropped to the image bounds.
    Half the overlap is discarded at internal tile edges. The detached output
    is assembled on the CPU, preserving the model's output dtype.
    """
    if X.ndim != 3 or any(size == 0 for size in X.shape):
        message = "X must be a non-empty tensor with shape (channels, height, width)"
        raise ValueError(message)

    chunk_size = 128
    height, width = X.shape[-2:]
    nruns = define_iteration(
        dimension=(height, width),
        chunk_size=chunk_size,
        overlap=overlap,
    )

    for index, (y, x) in enumerate(tqdm(nruns)):
        Xchunk = X[:, y : y + chunk_size, x : x + chunk_size]
        patch_height, patch_width = Xchunk.shape[-2:]
        if patch_height < chunk_size or patch_width < chunk_size:
            Xchunk = F.pad(
                Xchunk,
                (0, chunk_size - patch_width, 0, chunk_size - patch_height),
                mode="replicate",
            )

        prediction = model(Xchunk[None])
        if prediction.ndim != 4 or prediction.shape[0] != 1:
            message = "model must return a tensor with shape (1, channels, height, width)"
            raise ValueError(message)
        result = prediction[0]

        if index == 0:
            res_n, remainder = divmod(result.shape[1], chunk_size)
            if res_n < 1 or remainder or result.shape[2] != chunk_size * res_n:
                message = "model must use the same positive integer scale on both spatial axes"
                raise ValueError(message)
            output = torch.zeros(
                (result.shape[0], height * res_n, width * res_n),
                dtype=result.dtype,
                device="cpu",
            )
        elif result.shape != (output.shape[0], chunk_size * res_n, chunk_size * res_n):
            message = "model output channels and scale must be consistent across tiles"
            raise ValueError(message)

        skip = overlap * res_n // 2
        top = skip if y > 0 else 0
        left = skip if x > 0 else 0
        bottom = patch_height * res_n if y + chunk_size >= height else chunk_size * res_n - skip
        right = patch_width * res_n if x + chunk_size >= width else chunk_size * res_n - skip

        output[:, y * res_n + top : y * res_n + bottom, x * res_n + left : x * res_n + right] = (
            result[:, top:bottom, left:right].detach().cpu()
        )

    return output
