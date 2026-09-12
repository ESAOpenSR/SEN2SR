"""Core N-pass test-time-augmentation ensemble loop (Phase 5).

`run_tta_ensemble` runs the SAME frozen model once per transform in
``transforms``: the transform's `forward` is applied to the LR input, the
model runs unmodified (identical calling convention to every other model
call in this project -- `model(x[None]).squeeze(0)`), and the transform's
`inverse` is applied to the SR output before it is folded into a streaming
(Welford) mean/variance accumulator. Every prediction is therefore compared
in the SAME canonical orientation, so the resulting dispersion measures
genuine model disagreement about the scene, not a geometry mismatch.

This module never imports `sen2sr`/`mlstac` -- ``model`` is any callable
following that exact convention; the real model call lives in
`experiments/uncertainty/run_experiment.py`, matching every other phase's
separation of concerns.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch

from frame.uncertainty.errors import InvalidEnsembleConfigError, InvalidTransformError
from frame.uncertainty.statistics import WelfordAccumulator
from frame.uncertainty.transforms import Transform, validate_round_trip


@dataclass(frozen=True)
class EnsembleRunResult:
    mean_prediction: torch.Tensor  # (C, H*scale, W*scale)
    variance_prediction: torch.Tensor  # population variance, same shape
    std_prediction: torch.Tensor  # same shape
    n: int
    transform_names: Tuple[str, ...]
    per_member_predictions: Optional[Tuple[torch.Tensor, ...]]
    per_member_inference_seconds: Tuple[float, ...]
    total_seconds: float
    seed: int


def run_tta_ensemble(
    model,
    input_tensor: torch.Tensor,
    transforms: Sequence[Transform],
    *,
    seed: int = 42,
    keep_per_member_predictions: bool = True,
) -> EnsembleRunResult:
    """Run ``model`` once per transform in ``transforms`` and accumulate
    per-pixel mean/variance/std across the (de-transformed) predictions.

    Args:
        model: Callable following the project's standard convention --
            ``model(x[None]) -> y`` where ``y`` has a leading batch axis of
            size 1 (i.e. the same calling convention as every real
            `mlstac.load(...).compiled_model(...)` call elsewhere in this
            project). Never imported or constructed here.
        input_tensor: The LR input, shape (C, H, W).
        transforms: Non-empty sequence of `frame.uncertainty.transforms.Transform`.
            Each is validated (`validate_round_trip`) against `input_tensor`
            BEFORE any inference call, so a misconfigured transform fails
            immediately rather than after spending GPU time.
        seed: Recorded on the result and set via `torch.manual_seed` before
            the loop starts -- the geometric transforms themselves are
            already fully deterministic; this only matters if a transform
            with a stochastic component (e.g. an optional noise
            perturbation -- see frame/uncertainty/README.md) is included.
        keep_per_member_predictions: If True (default), every de-transformed
            per-member prediction is kept (needed to later identify which
            transform disagreed most -- see
            `frame.uncertainty.report.run_stochastic_uncertainty`). If
            False, only the running Welford accumulator is kept, which is
            the memory-frugal option for a large AOI.

    Returns:
        An EnsembleRunResult with the streaming mean/variance/std and
        per-member timing.
    """
    if input_tensor.ndim != 3:
        raise InvalidTransformError(
            f"Expected a 3-D (bands, H, W) input tensor, got {input_tensor.ndim}-D shape {tuple(input_tensor.shape)}."
        )
    if len(transforms) == 0:
        raise InvalidEnsembleConfigError("transforms must be a non-empty sequence (the identity transform alone is valid).")

    for t in transforms:
        validate_round_trip(t, input_tensor)

    torch.manual_seed(seed)

    accumulator = WelfordAccumulator()
    per_member_predictions = [] if keep_per_member_predictions else None
    per_member_seconds = []

    total_t0 = time.time()
    for t in transforms:
        x_t = t.forward(input_tensor)

        member_t0 = time.time()
        with torch.no_grad():
            y_t = model(x_t[None]).squeeze(0)
        if y_t.is_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - member_t0

        y_t = y_t.detach().cpu()
        y = t.inverse(y_t)

        accumulator.update(y)
        if keep_per_member_predictions:
            per_member_predictions.append(y)
        per_member_seconds.append(elapsed)
    total_seconds = time.time() - total_t0

    return EnsembleRunResult(
        mean_prediction=accumulator.mean,
        variance_prediction=accumulator.variance,
        std_prediction=accumulator.std,
        n=len(transforms),
        transform_names=tuple(t.name for t in transforms),
        per_member_predictions=tuple(per_member_predictions) if per_member_predictions is not None else None,
        per_member_inference_seconds=tuple(per_member_seconds),
        total_seconds=total_seconds,
        seed=seed,
    )
