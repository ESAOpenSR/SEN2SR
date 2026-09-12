"""Cross-tile overlap consistency check (docs/FRAME_TECHNICAL_SPEC.md Section 9.2).

For AOIs large enough to need `sen2sr.utils.predict_large`'s tiling, this
measures discrepancy between two neighboring tiles' SR outputs in the pixel
region where they overlap, before blending -- a large discrepancy there is a
seam-artifact signal.

VERIFIED LIMITATION -- read before trying to wire this to a real multi-tile
run: `sen2sr.utils.predict_large` (read in full; see `sen2sr/utils.py`)
writes each cropped tile directly into its own single shared ``output``
tensor and returns only that already-blended mosaic -- it does not return,
yield, or otherwise expose the raw per-tile SR arrays *before* they are
cropped and written. There is therefore no way to obtain "the two tiles'
outputs in their overlap region, before blending" from `predict_large` as it
exists today without modifying `sen2sr` itself, which is explicitly out of
scope for this project (docs/FRAME_TECHNICAL_SPEC.md Section 2.4). This
module is provided as a ready-to-use, pure function operating on an
*equivalent abstraction* -- two tile arrays plus their pixel-space placement
on a shared output grid -- of the kind a future FRAME-owned tiling
orchestrator (not yet built; see docs/FRAME_TECHNICAL_SPEC.md Section 10's
note that per-tile geolocation bookkeeping is FRAME's responsibility, not
upstream's) would already need to produce for georeferencing purposes. It is
exercised here only against synthetic tiles (frame/tests/test_consistency_tiles.py);
Baseline 0's real scene is a single 128x128 patch and never triggers
`predict_large`'s multi-tile path at all, so there is currently no real
multi-tile FRAME run to attach this to -- see experiments/consistency/README.md.

`ComputationStatus.INVALID_INPUT` (rather than a raised exception) is used
here -- not in the other frame.consistency modules -- specifically because
this function is meant to be called in a loop over many tile *pairs* from a
tiling scheme (e.g. every adjacent pair in a grid of tiles): a single pair
that turns out not to actually overlap, or that has mismatched band counts,
should not abort a whole-AOI diagnostic sweep the way a truly malformed
single-tensor input does elsewhere in this package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from frame.consistency.errors import ShapeMismatchError
from frame.consistency.status import ComputationStatus


@dataclass(frozen=True)
class TileOverlapResult:
    status: ComputationStatus
    overlap_shape: Optional[Tuple[int, int]]
    valid_pixel_count: int
    mean_abs_discrepancy: Optional[float]
    rmse: Optional[float]
    max_abs_discrepancy: Optional[float]


def compare_tile_overlap(
    tile_a: np.ndarray,
    offset_a: Tuple[int, int],
    tile_b: np.ndarray,
    offset_b: Tuple[int, int],
    *,
    mask_a: Optional[np.ndarray] = None,
    mask_b: Optional[np.ndarray] = None,
) -> TileOverlapResult:
    """Compare two SR tile outputs in their pixel-space overlap region.

    Args:
        tile_a, tile_b: (bands, h, w) arrays already placed on a shared
            output pixel grid (same CRS/grid, same pixel size -- e.g. two
            neighboring SR tiles from the same run).
        offset_a, offset_b: (row, col) pixel offset of each tile's top-left
            corner in that shared grid.
        mask_a, mask_b: optional (h, w) boolean validity masks matching each
            tile (see `frame.preprocessing.ValidityMask.array`) -- a pixel
            masked out in *either* tile is excluded, since comparing against
            a masked (not a real observation) pixel would measure noise, not
            seam consistency.

    Returns:
        A TileOverlapResult. ``status`` is ``INVALID_INPUT`` if the tiles, as
        placed, do not actually overlap or have different band counts;
        ``NOT_COMPUTABLE`` if they overlap but every pixel there is masked;
        ``COMPUTABLE`` otherwise.
    """
    if tile_a.ndim != 3 or tile_b.ndim != 3:
        raise ShapeMismatchError(
            f"Expected 3-D (bands, h, w) tile arrays, got shapes {tile_a.shape} and {tile_b.shape}."
        )

    if tile_a.shape[0] != tile_b.shape[0]:
        return TileOverlapResult(ComputationStatus.INVALID_INPUT, None, 0, None, None, None)

    ay0, ax0 = offset_a
    by0, bx0 = offset_b
    _, ah, aw = tile_a.shape
    _, bh, bw = tile_b.shape

    top = max(ay0, by0)
    left = max(ax0, bx0)
    bottom = min(ay0 + ah, by0 + bh)
    right = min(ax0 + aw, bx0 + bw)

    if bottom <= top or right <= left:
        return TileOverlapResult(ComputationStatus.INVALID_INPUT, None, 0, None, None, None)

    overlap_shape = (bottom - top, right - left)

    a_sub = tile_a[:, top - ay0 : bottom - ay0, left - ax0 : right - ax0]
    b_sub = tile_b[:, top - by0 : bottom - by0, left - bx0 : right - bx0]

    valid = np.ones(overlap_shape, dtype=bool)
    if mask_a is not None:
        valid &= np.asarray(mask_a, dtype=bool)[top - ay0 : bottom - ay0, left - ax0 : right - ax0]
    if mask_b is not None:
        valid &= np.asarray(mask_b, dtype=bool)[top - by0 : bottom - by0, left - bx0 : right - bx0]

    valid_pixel_count = int(valid.sum())
    if valid_pixel_count == 0:
        return TileOverlapResult(ComputationStatus.NOT_COMPUTABLE, overlap_shape, 0, None, None, None)

    diff = a_sub[:, valid].astype(np.float64) - b_sub[:, valid].astype(np.float64)
    abs_diff = np.abs(diff)
    return TileOverlapResult(
        ComputationStatus.COMPUTABLE,
        overlap_shape,
        valid_pixel_count,
        float(abs_diff.mean()),
        float(np.sqrt(np.mean(diff**2))),
        float(abs_diff.max()),
    )
