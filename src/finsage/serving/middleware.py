"""ASGI middleware for the FinSage-7B FastAPI service.

Provides request-ID tagging, structured JSON request logging (which never logs
filing text), baseline security headers, and in-memory rate limiting.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from finsage.logging_utils import get_logger
from finsage.serving.auth import get_api_key_from_headers, is_public_path
from finsage.serving.rate_limiter import InMemoryRateLimiter

logger = get_logger(__name__)

RequestResponseEndpoint = Callable[[Request], Awaitable[Response]]

#: Header used to propagate the per-request correlation ID.
REQUEST_ID_HEADER = "X-Request-ID"


def _client_id(request: Request) -> str:
    """Derive a stable rate-limit client id from the request.

    Prefers a hash of the API key; falls back to the client host. Hashing avoids
    storing or logging the raw key.

    Args:
        request: The incoming request.

    Returns:
        A short, opaque client identifier.
    """
    api_key = get_api_key_from_headers(request.headers)
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        return f"key:{digest[:16]}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Assign a UUID request id to each request and echo it in the response."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Attach a request id to ``request.state`` and the response headers.

        Args:
            request: The incoming request.
            call_next: The next handler in the chain.

        Returns:
            The downstream response with an ``X-Request-ID`` header.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Emit one structured JSON log line per request.

    Logs request metadata and the input character count only — never the filing
    excerpt content — unless ``log_request_body`` is explicitly enabled.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Log method/path/status/latency as JSON without leaking filing text.

        Args:
            request: The incoming request.
            call_next: The next handler in the chain.

        Returns:
            The downstream response.
        """
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        content_length = request.headers.get("content-length")
        record = {
            "request_id": getattr(request.state, "request_id", None),
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "input_chars": int(content_length) if content_length else None,
        }
        logger.info(json.dumps(record))
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add a baseline set of hardening response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Add security headers to every response.

        Args:
            request: The incoming request.
            call_next: The next handler in the chain.

        Returns:
            The downstream response with security headers set.
        """
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce per-client rate limits using an in-memory limiter.

    The limiter is resolved from ``request.app.state.rate_limiter`` so it can be
    swapped in tests; a fallback limiter is used if none is configured.

    Args:
        app: The ASGI app.
        fallback_limiter: Limiter used when the app has none configured.
    """

    def __init__(self, app: Callable, fallback_limiter: InMemoryRateLimiter | None = None) -> None:
        super().__init__(app)
        self._fallback = fallback_limiter or InMemoryRateLimiter()

    def _limiter(self, request: Request) -> InMemoryRateLimiter:
        """Return the active limiter for this request."""
        limiter = getattr(request.app.state, "rate_limiter", None)
        return limiter if isinstance(limiter, InMemoryRateLimiter) else self._fallback

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Allow or reject a request based on the per-client budget.

        Public paths (health/docs) are never rate limited so probes are not
        starved. Rejected requests return HTTP 429 with rate-limit headers.

        Args:
            request: The incoming request.
            call_next: The next handler in the chain.

        Returns:
            The downstream response, or a 429 response when over budget.
        """
        if is_public_path(request.url.path):
            return await call_next(request)

        allowed, meta = self._limiter(request).is_allowed(_client_id(request))
        headers = {
            "X-RateLimit-Limit": str(meta["limit"]),
            "X-RateLimit-Remaining": str(meta["remaining"]),
            "X-RateLimit-Reset": str(meta["reset_seconds"]),
        }
        if not allowed:
            request_id = getattr(request.state, "request_id", None)
            return JSONResponse(
                status_code=429,
                headers=headers,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": "Too many requests. Retry after the reset window.",
                    "request_id": request_id,
                },
            )
        response = await call_next(request)
        for name, value in headers.items():
            response.headers[name] = value
        return response


# Backwards-compatible alias for the Phase 1 logging middleware name.
RequestLoggingMiddleware = StructuredLoggingMiddleware
