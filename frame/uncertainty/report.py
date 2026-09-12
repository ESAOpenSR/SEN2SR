"""Top-level entry point: `run_stochastic_uncertainty` (Phase 5's desired
core API).

Ties `frame.uncertainty.ensemble.run_tta_ensemble` (the raw N-pass loop) to
`frame.uncertainty.statistics` (distribution summaries), and records enough
metadata to reproduce the run. This is the ONLY function most callers need
-- see `experiments/uncertainty/run_experiment.py` for the real integration
against the frozen `SEN2SRLite/NonReference_RGBN_x4` model.

What this estimates -- and what it does not
---------------------------------------------
This produces a **relative, architecture-conditioned model-stability proxy**:
how much the frozen model's own prediction moves when the same real
observation is presented to it in a different (but semantically identical)
geometric framing. It is NOT a calibrated probability that a pixel is
wrong, NOT a confidence interval, and NOT a physically rigorous uncertainty
bound -- there is no ground truth anywhere in this computation to calibrate
against. It is also NOT LAM (`sen2sr/xai/lam.py`): LAM is a gradient-based
explainability/sensitivity tool answering "which input pixels most
influence this output," computed via backprop against blurred copies of a
SINGLE input; this module answers a different question -- "how much does
the model's own prediction disagree with itself across equivalent views of
the same input" -- via repeated forward passes, no gradients, no blur ramp.
See frame/uncertainty/README.md for the full distinction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from frame.uncertainty.ensemble import run_tta_ensemble
from frame.uncertainty.errors import ShapeMismatchError
from frame.uncertainty.statistics import (
    UncertaintyDistributionStats,
    compute_distribution_stats,
)
from frame.uncertainty.transforms import DEFAULT_TRANSFORMS, Transform

SCALAR_SUMMARY_DEFINITION = (
    "Mean per-pixel standard deviation across the TTA ensemble, averaged over "
    "bands. A relative model-stability proxy, not a calibrated confidence value."
)


@dataclass(frozen=True)
class TransformDisagreement:
    transform_name: str
    inference_seconds: float
    mean_abs_deviation_from_ensemble_mean: Optional[float]


@dataclass(frozen=True)
class UncertaintyResult:
    mean_prediction: torch.Tensor
    std_prediction: torch.Tensor
    variance_prediction: torch.Tensor
    scalar_summary: float
    scalar_summary_definition: str
    overall_distribution: UncertaintyDistributionStats
    per_band_distribution: Dict[str, UncertaintyDistributionStats]
    per_transform_disagreement: Tuple[TransformDisagreement, ...]
    n: int
    transform_names: Tuple[str, ...]
    seed: int
    band_names: Tuple[str, ...]
    total_seconds: float
    metadata: Dict[str, Any]


def run_stochastic_uncertainty(
    model,
    input_tensor: torch.Tensor,
    *,
    transforms: Optional[Sequence[Transform]] = None,
    seed: int = 42,
    band_names: Sequence[str],
    keep_per_member_predictions: bool = True,
) -> UncertaintyResult:
    """Run stochastic test-time-augmentation uncertainty for one input.

    Args:
        model: Callable following the project's standard convention,
            `model(x[None]) -> y` (see `frame.uncertainty.ensemble`).
        input_tensor: The LR input, shape (C, H, W).
        transforms: The TTA transform set. Defaults to
            `frame.uncertainty.transforms.DEFAULT_TRANSFORMS` (the 6
            geometric symmetries) when not given.
        seed: Recorded and used to seed `torch.manual_seed` before the
            ensemble loop (see `run_tta_ensemble`).
        band_names: Name of each band along axis 0, in order -- must match
            `input_tensor`'s channel count.
        keep_per_member_predictions: If True (default), enables the
            per-transform disagreement breakdown; if False, only the
            streaming mean/variance/std are computed (lower memory).
    """
    if transforms is None:
        transforms = DEFAULT_TRANSFORMS

    if input_tensor.ndim == 3 and len(band_names) != input_tensor.shape[0]:
        raise ShapeMismatchError(
            f"band_names has {len(band_names)} entries but the input has {input_tensor.shape[0]} band(s)."
        )

    ensemble_result = run_tta_ensemble(
        model, input_tensor, transforms, seed=seed, keep_per_member_predictions=keep_per_member_predictions
    )

    overall_std_map = ensemble_result.std_prediction.mean(dim=0)  # (H, W), aggregated over bands
    overall_distribution = compute_distribution_stats(overall_std_map)
    per_band_distribution = {
        name: compute_distribution_stats(ensemble_result.std_prediction[i]) for i, name in enumerate(band_names)
    }

    per_transform_disagreement = []
    if ensemble_result.per_member_predictions is not None:
        for name, pred, secs in zip(
            ensemble_result.transform_names,
            ensemble_result.per_member_predictions,
            ensemble_result.per_member_inference_seconds,
        ):
            deviation = float((pred - ensemble_result.mean_prediction).abs().mean().item())
            per_transform_disagreement.append(TransformDisagreement(name, secs, deviation))
    else:
        for name, secs in zip(ensemble_result.transform_names, ensemble_result.per_member_inference_seconds):
            per_transform_disagreement.append(TransformDisagreement(name, secs, None))

    metadata: Dict[str, Any] = {
        "seed": seed,
        "n": ensemble_result.n,
        "transform_names": list(ensemble_result.transform_names),
        "band_names": list(band_names),
        "total_seconds": ensemble_result.total_seconds,
        "scalar_summary_definition": SCALAR_SUMMARY_DEFINITION,
    }

    return UncertaintyResult(
        mean_prediction=ensemble_result.mean_prediction,
        std_prediction=ensemble_result.std_prediction,
        variance_prediction=ensemble_result.variance_prediction,
        scalar_summary=overall_distribution.mean,
        scalar_summary_definition=SCALAR_SUMMARY_DEFINITION,
        overall_distribution=overall_distribution,
        per_band_distribution=per_band_distribution,
        per_transform_disagreement=tuple(per_transform_disagreement),
        n=ensemble_result.n,
        transform_names=ensemble_result.transform_names,
        seed=seed,
        band_names=tuple(band_names),
        total_seconds=ensemble_result.total_seconds,
        metadata=metadata,
    )
