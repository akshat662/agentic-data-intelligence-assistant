"""FastAPI application factory. Thin interface layer only -- see `adia.api.service` for the
orchestration logic and `adia.api.routes` for the endpoint definitions.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .routes import router

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the ADIA FastAPI application."""
    app = FastAPI(
        title="ADIA API",
        description="Thin HTTP interface over the ADIA agentic investigation graph.",
        version="0.1.0",
    )
    app.include_router(router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all so an unexpected failure (e.g. an unreachable LLM, a bad graph state)
        never leaks an internal stack trace or exception message to the client.
        """
        logger.exception("Unhandled error while handling %s %s", request.method, request.url)
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    return app


app = create_app()
