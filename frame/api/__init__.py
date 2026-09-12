"""FRAME API (Phase 7) -- a FastAPI backend exposing the tested Phase 1-6
FRAME pipeline (preprocessing -> geospatial -> frozen SEN2SR + uncertainty
-> self-consistency -> NDVI analysis) over HTTP for the SIH prototype.

This package only orchestrates the existing, already-tested `frame.*`
packages (see `frame.api.services.pipeline`); it never reimplements their
logic and never modifies `sen2sr/`. See `frame/api/README.md` for the full
endpoint reference.
"""

from __future__ import annotations

from frame.api.app import create_app

__all__ = ["create_app"]
