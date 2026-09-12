"""Streaming statistics for the TTA uncertainty ensemble (Phase 5).

Three independent pieces:

1. `WelfordAccumulator` -- Welford's online algorithm for mean/variance,
   updated one ensemble member at a time so the full set of N predictions
   never needs to be held in memory simultaneously (docs/FRAME_TECHNICAL_SPEC.md
   Section 13, and this phase's explicit "streaming accumulation is
   preferred" instruction). Reports **population** variance (divide by N,
   not N-1) -- the ensemble members here are the entire population of
   interest (this specific set of geometric views), not a sample drawn from
   a larger population.
2. `compute_distribution_stats` -- summarizes a per-pixel uncertainty map
   (e.g. the std map) as a distribution: mean, median, std-of-the-map,
   p90, p95, max, min. No single number here is privileged as "the"
   uncertainty score without saying which one; no threshold is applied to
   any of them.
3. `normalize_for_visualization` -- percentile-clip-and-rescale to [0, 1]
   for display ONLY. This never touches, and is never used to compute, any
   of the scientific values above -- see its own docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from frame.uncertainty.errors import ShapeMismatchError


class WelfordAccumulator:
    """Online (streaming) mean/population-variance accumulator.

    Call `update(x)` once per ensemble member, in any order -- the running
    `mean`/`variance`/`std` after each call reflect every member seen so
    far, without ever needing to store the members themselves.
    """

    def __init__(self) -> None:
        self._count = 0
        self._mean: Optional[torch.Tensor] = None
        self._m2: Optional[torch.Tensor] = None

    @property
    def count(self) -> int:
        return self._count

    def update(self, x: torch.Tensor) -> None:
        x = x.detach().to(torch.float64)
        if self._mean is None:
            self._mean = x.clone()
            self._m2 = torch.zeros_like(x)
            self._count = 1
            return
        if x.shape != self._mean.shape:
            raise ShapeMismatchError(
                f"Update shape {tuple(x.shape)} does not match the running accumulator shape {tuple(self._mean.shape)}."
            )
        self._count += 1
        delta = x - self._mean
        self._mean = self._mean + delta / self._count
        delta2 = x - self._mean
        self._m2 = self._m2 + delta * delta2

    @property
    def mean(self) -> torch.Tensor:
        if self._mean is None:
            raise ShapeMismatchError("No updates yet -- mean is undefined for an empty accumulator.")
        return self._mean.to(torch.float32)

    @property
    def variance(self) -> torch.Tensor:
        """Population variance (M2 / N)."""
        if self._mean is None:
            raise ShapeMismatchError("No updates yet -- variance is undefined for an empty accumulator.")
        if self._count < 1:
            raise ShapeMismatchError("No updates yet -- variance is undefined for an empty accumulator.")
        return (self._m2 / self._count).to(torch.float32)

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.variance.clamp(min=0.0))


@dataclass(frozen=True)
class UncertaintyDistributionStats:
    """Distribution of a per-pixel uncertainty map's values -- e.g. "what does
    the map of per-pixel standard deviations itself look like, statistically."
    No field here is a threshold; none should be read as a pass/fail cutoff.
    """

    mean: float
    median: float
    std: float
    p90: float
    p95: float
    min: float
    max: float
    n_pixels: int


def compute_distribution_stats(values: torch.Tensor) -> UncertaintyDistributionStats:
    """Summarize an arbitrary-shape tensor of uncertainty values as a distribution."""
    flat = values.detach().to(torch.float64).flatten().cpu().numpy()
    return UncertaintyDistributionStats(
        mean=float(np.mean(flat)),
        median=float(np.median(flat)),
        std=float(np.std(flat)),
        p90=float(np.percentile(flat, 90)),
        p95=float(np.percentile(flat, 95)),
        min=float(np.min(flat)),
        max=float(np.max(flat)),
        n_pixels=int(flat.size),
    )


def normalize_for_visualization(
    values: torch.Tensor, *, low_percentile: float = 2.0, high_percentile: float = 98.0
) -> torch.Tensor:
    """Percentile-clip and rescale ``values`` to [0, 1] for DISPLAY ONLY.

    This is a visualization convenience, not a scientific transformation --
    it is never applied before computing mean/variance/std or any
    distribution statistic above, and the returned tensor must never be
    saved as, or confused with, the actual uncertainty values (see
    frame/uncertainty/README.md's explicit statement on this). Does not
    mutate ``values``.
    """
    flat = values.detach().to(torch.float64)
    lo = torch.quantile(flat, low_percentile / 100.0)
    hi = torch.quantile(flat, high_percentile / 100.0)
    denom = (hi - lo).item()
    if denom <= 0:
        # a constant (or degenerate) map -- avoid a division by zero;
        # every pixel is equally "in the middle" of a non-existent range.
        return torch.full_like(flat, 0.5, dtype=torch.float32)
    normalized = ((flat - lo) / denom).clamp(0.0, 1.0)
    return normalized.to(torch.float32)
