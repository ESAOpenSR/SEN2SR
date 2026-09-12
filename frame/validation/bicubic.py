"""The naive baseline every SR result in this project is compared against.

Bicubic-with-antialiasing upsampling, matching the exact convention already
used for visualization in `experiments/baseline/run_baseline.py` and
`experiments/consistency/run_experiment.py` -- the standard "no learned
model" reference point in the SR literature, and the honest lower bound
Section 11's validation strategy compares SEN2SRLite against.

This is a different resampling operation from both the upstream Fourier
hard constraint's own LR-upsampling step (`sen2sr/models/tricks.py`, also
bicubic+antialias, but used internally to build the low-frequency
constraint, not as an evaluation baseline) and
`frame.consistency.downsample_to_lr_grid`'s area-average pooling (the
opposite direction: SR -> LR, not LR -> HR).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frame.validation.errors import ShapeMismatchError


def bicubic_upsample(lr: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Upsample ``lr`` (bands, H, W) to (bands, H*scale_factor, W*scale_factor)
    via bicubic interpolation with antialiasing.

    Args:
        lr: LR tensor, shape (C, H, W).
        scale_factor: Positive integer upscaling factor.

    Returns:
        A new float32 tensor, shape (C, H*scale_factor, W*scale_factor).
    """
    if lr.ndim != 3:
        raise ShapeMismatchError(
            f"Expected a 3-D (bands, height, width) tensor, got {lr.ndim}-D with shape {tuple(lr.shape)}."
        )
    if scale_factor < 1:
        raise ShapeMismatchError(f"scale_factor must be a positive integer, got {scale_factor!r}.")

    upsampled = F.interpolate(
        lr.float()[None], scale_factor=scale_factor, mode="bicubic", antialias=True
    )
    return upsampled[0].clamp(min=0.0)
