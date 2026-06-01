"""FastAPI application factory for FinSage-7B.

The module-level ``app`` is the ASGI target used by uvicorn
(``uvicorn finsage.serving.app:app``). The app imports and starts cleanly even
when the vLLM backend is offline; backend status is surfaced via ``/v1/ready``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finsage import __version__
from finsage.config import get_settings
from finsage.logging_utils import get_logger, setup_logging
from finsage.serving.errors import register_exception_handlers
from finsage.serving.middleware import (
    RateLimitMiddleware,
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    StructuredLoggingMiddleware,
)
from finsage.serving.rate_limiter import build_rate_limiter
from finsage.serving.routes import router

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FinSage-7B FastAPI application.

    Wires in the request-id, logging, security-header, rate-limit, and CORS
    middleware, registers the exception handlers, and mounts the ``/v1`` routes.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    if not settings.is_production and settings.api_secret_key in ("", "change-me"):
        logger.warning(
            "Running in development mode with a placeholder API_SECRET_KEY. "
            "Set ENVIRONMENT=production and a strong API_SECRET_KEY before deploying."
        )

    app = FastAPI(
        title="FinSage-7B API",
        version=__version__,
        description=(
            "Financial-filing analysis API in front of an internal vLLM server. "
            "Not financial advice — verify all outputs against the original filings."
        ),
    )

    # Shared rate limiter (in-memory or Redis per settings). Stored on app.state
    # so it can be swapped/inspected in tests.
    app.state.rate_limiter = build_rate_limiter(settings)

    # Middleware: the last added runs first (outermost). Order on a request is
    # RequestID -> Logging -> SecurityHeaders -> RateLimit -> CORS -> routes,
    # so every response (including 429s) is tagged, secured, and logged.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, fallback_limiter=app.state.rate_limiter)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(StructuredLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)
    app.include_router(router)
    return app


app = create_app()
