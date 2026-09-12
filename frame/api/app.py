"""FastAPI application factory for the FRAME API (Phase 7).

`create_app()` wires together the router (frame.api.routes), CORS (from
frame.api.config, development-friendly and explicit rather than "*"), and a
single generic exception handler. That handler is the ONE place any
exception -- an `frame.api.errors.APIError`, any `frame.*` package error,
or a genuine internal failure -- is turned into an HTTP response: it never
leaks a Python traceback to the caller, and 5xx failures are logged
server-side for debugging instead.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from frame.analysis.errors import AnalysisError
from frame.consistency.errors import ConsistencyError
from frame.geospatial.errors import GeospatialError
from frame.preprocessing.errors import PreprocessingError
from frame.uncertainty.errors import UncertaintyError

from frame.api import config
from frame.api.errors import APIError, status_code_for

logger = logging.getLogger("frame.api")


def create_app() -> FastAPI:
    app = FastAPI(title="FRAME API", version=config.API_VERSION)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from frame.api.routes import router

    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "request_validation_error", "code": "request_validation_error", "detail": str(exc)},
        )

    def _known_error_response(request: Request, exc: Exception) -> JSONResponse:
        # Every case reachable here is a known 4xx (an APIError subclass, or
        # one of the frame.* package's own input-validation error families
        # -- see frame.api.errors.status_code_for). Genuine 5xx failures are
        # NOT registered here; they fall through to Starlette's default
        # server-error handling below, which never echoes a traceback to
        # the caller.
        status_code = status_code_for(exc)
        code = exc.code if isinstance(exc, APIError) else "frame_error"
        return JSONResponse(status_code=status_code, content={"error": code, "code": code, "detail": str(exc)})

    for exc_type in (APIError, PreprocessingError, GeospatialError, ConsistencyError, UncertaintyError, AnalysisError):
        app.add_exception_handler(exc_type, _known_error_response)

    return app


app = create_app()
