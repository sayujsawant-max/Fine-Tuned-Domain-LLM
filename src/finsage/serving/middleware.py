"""Request-logging middleware for the FastAPI service.

Authentication and richer safety checks are layered in during Phase 9; Phase 1
ships structured request logging so the service is observable from the start.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from finsage.logging_utils import get_logger

logger = get_logger(__name__)

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log each request's method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process a request, logging method, path, status, and duration.

        Args:
            request: The incoming request.
            call_next: The next handler in the middleware chain.

        Returns:
            The downstream response.
        """
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
