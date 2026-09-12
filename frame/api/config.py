"""Environment-driven configuration for the FRAME API (Phase 7).

Every setting here has a sensible default and can be overridden by an
environment variable -- no absolute developer-machine path is hard-coded.
Kept deliberately small: this is a prototype/demo backend, not a
production configuration system.
"""

from __future__ import annotations

import os
from pathlib import Path

API_VERSION = "0.1.0"

#: Where uploaded rasters and generated job artifacts (GeoTIFFs, tensors,
#: JSON) are written. Explicitly NOT inside any source-code directory --
#: defaults to a per-user cache location, matching the convention every
#: prior phase's experiment scripts already use for their own caches
#: (``~/.cache/sen2sr_baseline``, ``~/.config/opensr_test``, ...).
WORKSPACE_DIR = Path(
    os.environ.get("FRAME_API_WORKSPACE_DIR", str(Path.home() / ".cache" / "frame_api" / "workspace"))
)

#: The SEN2SRLite/NonReference_RGBN_x4 weights cache. Reuses the EXACT
#: same environment variable every experiment script from Phase 0 onward
#: already reads, so the API picks up an already-downloaded cache with no
#: new download and no new env var to configure.
MODEL_WEIGHTS_CACHE_DIR = Path(
    os.environ.get(
        "SEN2SR_BASELINE_WEIGHTS_DIR",
        str(Path.home() / ".cache" / "sen2sr_baseline" / "SEN2SRLite_RGBN"),
    )
)

MODEL_MANIFEST_URL = (
    "https://huggingface.co/tacofoundation/sen2sr/resolve/main/"
    "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
)
MODEL_NAME = "SEN2SRLite/NonReference_RGBN_x4"

#: "auto" resolves to cuda if available, else cpu, at request time --
#: matching every prior phase's `torch.cuda.is_available()` check. Set to
#: "cpu" or "cuda" explicitly to override.
DEVICE = os.environ.get("FRAME_API_DEVICE", "auto")

#: Default TTA ensemble seed (frame.uncertainty), matching every prior
#: phase's default of 42.
UNCERTAINTY_SEED = int(os.environ.get("FRAME_API_UNCERTAINTY_SEED", "42"))

#: Comma-separated list of allowed CORS origins for local frontend
#: development. Explicit, not a wildcard, by default.
CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "FRAME_API_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]


def ensure_workspace_dirs() -> None:
    (WORKSPACE_DIR / "uploads").mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "jobs").mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "analyses").mkdir(parents=True, exist_ok=True)
