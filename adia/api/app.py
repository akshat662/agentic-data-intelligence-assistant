"""FastAPI application factory. Thin interface layer only -- see `adia.api.service` for the
orchestration logic and `adia.api.routes` for the endpoint definitions.
"""

import logging
import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router

logger = logging.getLogger(__name__)

#: Origins allowed to call this API from a browser (e.g. the Next.js dev server). Deliberately
#: an explicit allowlist, never "*" -- `ADIA_CORS_ORIGINS` overrides it with a comma-separated
#: list for other environments (e.g. a deployed frontend origin).
_DEFAULT_CORS_ORIGINS = ["http://localhost:3000"]


def _cors_origins() -> list[str]:
    raw = os.environ.get("ADIA_CORS_ORIGINS", "").strip()
    if not raw:
        return _DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> FastAPI:
    """Build the ADIA FastAPI application."""
    app = FastAPI(
        title="ADIA API",
        description="Thin HTTP interface over the ADIA agentic investigation graph.",
        version="0.1.0",
    )
    app.include_router(router)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all so an unexpected failure (e.g. an unreachable LLM, a bad graph state)
        never leaks an internal stack trace or exception message to the client.
        """
        logger.exception("Unhandled error while handling %s %s", request.method, request.url)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
