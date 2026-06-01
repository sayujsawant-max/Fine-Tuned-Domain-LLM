"""Custom exceptions and FastAPI exception handlers for the FinSage-7B API.

Every error response is normalised to the :class:`ErrorResponse` shape
(``error`` / ``detail`` / ``request_id``) so clients can rely on a single
contract regardless of where the failure originated.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from finsage.logging_utils import get_logger

logger = get_logger(__name__)


class FinsageAPIError(Exception):
    """Base class for FinSage API errors.

    Attributes:
        message: Human-readable error message.
        status_code: HTTP status code to return.
        error: Short error category string.
    """

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error: str = "internal_error"

    def __init__(self, message: str) -> None:
        """Initialise the error.

        Args:
            message: Human-readable detail for the response body.
        """
        super().__init__(message)
        self.message = message


class VLLMUnavailableError(FinsageAPIError):
    """Raised when the vLLM backend cannot be reached (HTTP 503)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error = "vllm_unavailable"


class AuthenticationError(FinsageAPIError):
    """Raised when API key authentication fails (HTTP 401)."""

    status_code = status.HTTP_401_UNAUTHORIZED
    error = "authentication_error"


class RateLimitError(FinsageAPIError):
    """Raised when a client exceeds its rate limit (HTTP 429)."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    error = "rate_limit_exceeded"


def _request_id(request: Request) -> str | None:
    """Return the request_id stashed on ``request.state`` if present."""
    return getattr(request.state, "request_id", None)


def _error_payload(error: str, detail: str | None, request_id: str | None) -> dict[str, str | None]:
    """Build a JSON-serialisable error body."""
    return {"error": error, "detail": detail, "request_id": request_id}


async def finsage_error_handler(request: Request, exc: FinsageAPIError) -> JSONResponse:
    """Handle :class:`FinsageAPIError` and subclasses.

    Args:
        request: The incoming request.
        exc: The raised FinSage error.

    Returns:
        A JSON error response with the error's status code.
    """
    request_id = _request_id(request)
    if exc.status_code >= 500:
        logger.error("%s: %s (request_id=%s)", exc.error, exc.message, request_id)
    else:
        logger.info("%s: %s (request_id=%s)", exc.error, exc.message, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(exc.error, exc.message, request_id),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request schema validation errors (HTTP 422).

    Args:
        request: The incoming request.
        exc: The validation error raised by FastAPI.

    Returns:
        A JSON error response describing the first validation failure.
    """
    errors = exc.errors()
    detail = (
        "; ".join(
            f"{'.'.join(str(loc) for loc in e.get('loc', []))}: {e.get('msg', '')}".strip(": ")
            for e in errors
        )
        or "Request validation failed."
    )
    return JSONResponse(
        status_code=422,
        content=_error_payload("validation_error", detail, _request_id(request)),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle Starlette/FastAPI HTTP exceptions into the uniform shape.

    Args:
        request: The incoming request.
        exc: The HTTP exception.

    Returns:
        A JSON error response preserving the original status code.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload("http_error", str(exc.detail), _request_id(request)),
    )


async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle any uncaught exception as HTTP 500.

    Args:
        request: The incoming request.
        exc: The uncaught exception.

    Returns:
        A generic JSON 500 response (no internal detail leaked).
    """
    request_id = _request_id(request)
    logger.exception("Unhandled error (request_id=%s): %s", request_id, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload("internal_error", "An unexpected error occurred.", request_id),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all FinSage exception handlers on ``app``.

    Args:
        app: The FastAPI application to configure.
    """
    app.add_exception_handler(FinsageAPIError, finsage_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unexpected_error_handler)
