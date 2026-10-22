import torch
import itertools
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

    if chunk_size > max(dimx, dimy):
        return [(0, 0)]

    # Adjust step to create overlap
    y_step = chunk_size - overlap
    x_step = chunk_size - overlap

    # Generate initial chunk positions
    iterchunks = list(itertools.product(range(0, dimy, y_step), range(0, dimx, x_step)))

    # Fix chunks at the edges to stay within bounds
    iterchunks_fixed = fix_lastchunk(
        iterchunks=iterchunks, s2dim=dimension, chunk_size=chunk_size
    )

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
    
    # Run always in patches of 128x128 with 32 of overlap
    nruns = define_iteration(
        dimension=(X.shape[1], X.shape[2]),
        chunk_size=128,
        overlap=overlap,
    )
    
    # Define the output metadata        
    for index, point in enumerate(tqdm(nruns)):        
        
        # Read a block of the image        
        Xchunk = X[:, point[0] : (point[0] + 128), point[1] : (point[1] + 128)]
        
        # Predict the SR
        result = model(Xchunk[None]).squeeze(0)
        
        # If index is 0, create the output image
        if index == 0:
            res_n = result.shape[1] // 128
            output = torch.zeros(
                (result.shape[0], X.shape[1] * res_n, X.shape[2] * res_n),
                dtype=result.dtype,
                device="cpu",
            )
            

        # Define the offset in the output space
        # If the point is at the border, the offset is 0 
        # otherwise consider the overlap
        offset_y = point[0] * res_n + overlap * res_n // 2
        offset_x = point[1] * res_n + overlap * res_n // 2
        if point[0] == 0:
            offset_y = 0
        if point[1] == 0:
            offset_x = 0        
            
        
        
        # Our output is always 128*res_n x 128*res_n, 
        # Crop this batch output in order to fit in the 
        # output image
        #         
        # There is three conditions:
        #  - The patch is at the initial borders        
        #  - The patch is at the final borders
        #  - The patch is in the middle of the image
        skip = overlap * res_n // 2

        # Work in the X axis
        if offset_x == 0 and point[1] + 128 < X.shape[2]: # Initial border
            length_x = 128 * res_n - skip
            result = result[:, :, :length_x]
        elif (offset_x + 128) == X.shape[2]:
            length_x = 128 * res_n
            result = result[:, :, :length_x]
        elif offset_x == 0:
            length_x = 128 * res_n
        else:
            skip = overlap * res_n // 2
            length_x = 128 * res_n - skip
            result = result[:, :, skip:(128 * res_n)]
            
        # Work in the Y axis
        if offset_y == 0 and point[0] + 128 < X.shape[1]:            
            length_y = 128 * res_n - skip
            result = result[:, :length_y, :]            
        elif (offset_y + 128) == X.shape[1]:
            length_y = 128 * res_n
            result = result[:, :length_y, :]  
        elif offset_y == 0:
            length_y = 128 * res_n          
        else:
            length_y = 128 * res_n - skip
            result = result[:, skip:(128 * res_n), :]
        
        # Write the result in the output image
        output[:, offset_y:(offset_y + length_y), offset_x:(offset_x + length_x)] = result.detach().cpu()

    return output
        
