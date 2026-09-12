"""Cached model loading for the FRAME API (Phase 7).

The SEN2SRLite/NonReference_RGBN_x4 weights are downloaded/loaded at most
once per device per running API process -- not on every request -- via a
simple module-level cache keyed by device string, matching this phase's
explicit "keep model initialization cached" instruction.

`ensure_weights`/`load_compiled_model` are injectable so this module is
testable without touching the network or the real model
(`frame/tests/test_api_model_service.py` does exactly this); the defaults
here mirror the identical `mlstac`-based pattern every prior phase's
experiment scripts already use, not a new loading mechanism.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from frame.api import config

_model_cache: Dict[str, Any] = {}


def _default_ensure_weights(cache_dir: Path) -> Path:
    manifest_path = cache_dir / "mlm.json"
    if manifest_path.exists():
        return cache_dir
    import mlstac

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    mlstac.download(file=config.MODEL_MANIFEST_URL, output_dir=str(cache_dir))
    return cache_dir


def _default_load_compiled_model(weight_dir: Path, device: str) -> Any:
    import mlstac

    return mlstac.load(str(weight_dir)).compiled_model(device=device)


def resolve_device(device: str) -> str:
    """"auto" -> cuda if available, else cpu; any other value is returned unchanged."""
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(
    device: Optional[str] = None,
    *,
    ensure_weights: Callable[[Path], Path] = _default_ensure_weights,
    load_compiled_model: Callable[[Path, str], Any] = _default_load_compiled_model,
) -> Any:
    """Return the (cached) compiled SEN2SRLite/NonReference_RGBN_x4 model
    for ``device``, loading it only on the first call for that device."""
    resolved_device = resolve_device(device or config.DEVICE)
    if resolved_device not in _model_cache:
        weight_dir = ensure_weights(config.MODEL_WEIGHTS_CACHE_DIR)
        _model_cache[resolved_device] = load_compiled_model(weight_dir, resolved_device)
    return _model_cache[resolved_device]


def clear_cache() -> None:
    """Drop every cached model -- used by tests, and safe to call anytime."""
    _model_cache.clear()
