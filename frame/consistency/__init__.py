"""FRAME spectral-consistency diagnostics layer.

Independent, POST-HOC measurements of how well an SR output remains
consistent with the original Sentinel-2 observation it was derived from.
This package does not change what the model outputs -- the upstream Fourier
hard constraint (`sen2sr/models/tricks.py`, unmodified) already does that,
structurally, on every inference. This package only *measures* the result,
after the fact, with numbers that can be reported, logged, and inspected.

See docs/FRAME_TECHNICAL_SPEC.md Section 9.2 for the design this implements,
and README.md in this package for the concrete contract, the exact
mathematical definitions, and why no numeric pass/fail threshold is chosen
here.

Public API:
    run_consistency_diagnostics, ConsistencyDiagnostics   (report.py)
    compute_downsample_consistency, downsample_consistency_from_arrays,
        DownsampleConsistencyResult, BandDiscrepancy       (band_discrepancy.py)
    downsample_to_lr_grid, AREA_AVERAGE_POOL               (downsample.py)
    compute_ndvi, compute_simple_ratio,
    compute_ndvi_comparison, compute_b08_b04_ratio_comparison,
        SpectralIndexComparison, NDVI, B08_B04_RATIO       (spectral_ratios.py)
    compare_tile_overlap, TileOverlapResult                (tiles.py)
    ComputationStatus                                      (status.py)
    Exceptions: ConsistencyError and its subclasses (see .errors)
"""

from frame.consistency.band_discrepancy import (
    BandDiscrepancy,
    DownsampleConsistencyResult,
    compute_downsample_consistency,
    downsample_consistency_from_arrays,
)
from frame.consistency.downsample import AREA_AVERAGE_POOL, downsample_to_lr_grid
from frame.consistency.errors import (
    ConsistencyError,
    InvalidMaskError,
    MissingBandError,
    ScaleFactorError,
    ShapeMismatchError,
)
from frame.consistency.report import ConsistencyDiagnostics, run_consistency_diagnostics
from frame.consistency.spectral_ratios import (
    B08_B04_RATIO,
    NDVI,
    SpectralIndexComparison,
    compute_b08_b04_ratio_comparison,
    compute_ndvi,
    compute_ndvi_comparison,
    compute_simple_ratio,
)
from frame.consistency.status import ComputationStatus
from frame.consistency.tiles import TileOverlapResult, compare_tile_overlap

__all__ = [
    "run_consistency_diagnostics",
    "ConsistencyDiagnostics",
    "compute_downsample_consistency",
    "downsample_consistency_from_arrays",
    "DownsampleConsistencyResult",
    "BandDiscrepancy",
    "downsample_to_lr_grid",
    "AREA_AVERAGE_POOL",
    "compute_ndvi",
    "compute_simple_ratio",
    "compute_ndvi_comparison",
    "compute_b08_b04_ratio_comparison",
    "SpectralIndexComparison",
    "NDVI",
    "B08_B04_RATIO",
    "compare_tile_overlap",
    "TileOverlapResult",
    "ComputationStatus",
    "ConsistencyError",
    "ShapeMismatchError",
    "ScaleFactorError",
    "InvalidMaskError",
    "MissingBandError",
]
