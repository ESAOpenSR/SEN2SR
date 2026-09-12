"""Deterministic SR -> LR grid reduction for self-consistency diagnostics.

Method: non-overlapping area-average pooling (`torch.nn.functional.avg_pool2d`
with `kernel_size == stride == scale_factor`). Each output (LR-grid) pixel is
the arithmetic mean of the `scale_factor x scale_factor` block of SR pixels
it corresponds to.

Why this method, specifically, and not something else:

- It is the literal inverse of "one LR pixel's area expanded into a
  scale_factor x scale_factor block of SR pixels" -- the same physical
  picture (area-integrated sensor response) that motivates comparing an SR
  output back against its real LR observation in the first place.
- It has no free parameters (no kernel choice, no antialiasing flag), so
  re-running this diagnostic on the same tensors always reproduces the exact
  same number -- important for a *diagnostic*, where the point is measuring
  discrepancy, not producing the visually smoothest downsample.
- It is deliberately NOT the same operation as the upstream Fourier hard
  constraint's own resampling step (`sen2sr/models/tricks.py`, which
  bicubic-upsamples the LR input to the SR size, with antialiasing, before
  an FFT-domain recombination). That step runs in the opposite direction
  (LR -> SR-sized) and serves a different purpose (constraining what the
  model is allowed to output), not measuring it after the fact. Reusing its
  exact bicubic kernel here would conflate "what the model was constrained
  to do" with "an independent check that it actually did it" -- see
  frame/consistency/README.md for the full relationship to that constraint.

This module never fabricates a result for a bad shape: an SR tensor whose
spatial dimensions are not evenly divisible by `scale_factor` is *rejected*,
not silently cropped or padded, since that would quietly discard the exact
information (which SR pixels belong to which LR pixel's block) this
diagnostic depends on to line up correctly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from frame.consistency.errors import ScaleFactorError, ShapeMismatchError

AREA_AVERAGE_POOL = "area_average_pool"


def downsample_to_lr_grid(sr: torch.Tensor, scale_factor: int | float) -> torch.Tensor:
    """Reduce ``sr`` (bands, H, W) to (bands, H/scale_factor, W/scale_factor)
    via non-overlapping area-average pooling.

    Args:
        sr: SR tensor, shape (C, H, W).
        scale_factor: The SR model's upscaling factor (e.g. 4 for the proven
            RGBN path). Must be a positive integer (or an integral float,
            e.g. ``4.0``); H and W must each be evenly divisible by it.

    Returns:
        A new tensor, shape (C, H // scale_factor, W // scale_factor),
        float32.
    """
    if sr.ndim != 3:
        raise ShapeMismatchError(
            f"Expected a 3-D (bands, height, width) tensor, got {sr.ndim}-D with shape {tuple(sr.shape)}."
        )

    if scale_factor < 1 or int(scale_factor) != scale_factor:
        raise ScaleFactorError(
            f"scale_factor must be a positive integer, got {scale_factor!r}."
        )
    scale_factor = int(scale_factor)

    _, height, width = sr.shape
    if height % scale_factor != 0 or width % scale_factor != 0:
        raise ShapeMismatchError(
            f"SR shape ({height}, {width}) is not evenly divisible by scale_factor={scale_factor}; "
            "refusing to silently crop or pad."
        )

    pooled = F.avg_pool2d(sr.float()[None], kernel_size=scale_factor, stride=scale_factor)
    return pooled[0]
