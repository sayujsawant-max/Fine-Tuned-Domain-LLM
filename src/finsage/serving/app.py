"""FastAPI application factory for FinSage-7B.

Exposes ``GET /v1/health`` and ``POST /v1/chat``. The module-level ``app`` is
the ASGI target used by uvicorn (``uvicorn finsage.serving.app:app``).
"""

from __future__ import annotations

from fastapi import FastAPI

from finsage import __version__
from finsage.config import get_settings
from finsage.logging_utils import setup_logging
from finsage.serving.middleware import RequestLoggingMiddleware
from finsage.serving.routes import router


def create_app() -> FastAPI:
    """Build and configure the FinSage-7B FastAPI application.

    Returns:
        A configured :class:`fastapi.FastAPI` instance with logging middleware
        and the ``/v1`` routes mounted.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title="FinSage-7B API",
        version=__version__,
        description=(
            "Financial-filing analysis API. Not financial advice — verify all "
            "outputs against the original filings."
        ),
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.include_router(router)
    return app


app = create_app()
