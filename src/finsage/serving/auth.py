"""API key authentication for the FinSage-7B service.

Keys may be supplied via the ``X-API-Key`` header or an
``Authorization: Bearer <key>`` header. Comparisons use
:func:`secrets.compare_digest` to resist timing attacks. Public paths
(``/v1/health``, docs) are exempt.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping

from fastapi import Request

from finsage.config import Settings, get_settings
from finsage.logging_utils import get_logger
from finsage.serving.errors import AuthenticationError

logger = get_logger(__name__)

#: Placeholder secret that must never be used in production.
PLACEHOLDER_KEY = "change-me"

#: Paths that never require authentication.
PUBLIC_PATHS: frozenset[str] = frozenset({"/v1/health", "/docs", "/openapi.json", "/redoc"})


def is_public_path(path: str) -> bool:
    """Return whether ``path`` is exempt from authentication.

    Args:
        path: The request path (without query string).

    Returns:
        ``True`` for health/docs paths, ``False`` otherwise.
    """
    normalised = path.rstrip("/") or "/"
    if normalised in PUBLIC_PATHS or path in PUBLIC_PATHS:
        return True
    # Swagger UI static assets live under /docs/...
    return normalised.startswith("/docs")


def get_api_key_from_headers(headers: Mapping[str, str]) -> str | None:
    """Extract an API key from request headers.

    Checks ``X-API-Key`` first, then an ``Authorization: Bearer <key>`` header.

    Args:
        headers: A case-insensitive header mapping (e.g. ``request.headers``).

    Returns:
        The supplied key, or ``None`` if no key header is present.
    """
    api_key = headers.get("x-api-key") or headers.get("X-API-Key")
    if api_key:
        return api_key.strip()

    authorization = headers.get("authorization") or headers.get("Authorization")
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[len("bearer ") :].strip() or None
    return None


def verify_api_key(provided_key: str | None, expected_key: str | None) -> bool:
    """Securely compare a provided key against the expected key.

    Args:
        provided_key: The key supplied by the client, if any.
        expected_key: The configured server secret, if any.

    Returns:
        ``True`` if both keys are present and equal (constant-time compare).
    """
    if not provided_key or not expected_key:
        return False
    return secrets.compare_digest(provided_key, expected_key)


def _secret_is_configured(settings: Settings) -> bool:
    """Return whether a real (non-placeholder) secret is configured."""
    key = settings.api_secret_key
    return bool(key) and key != PLACEHOLDER_KEY


async def require_api_key(request: Request) -> str | None:
    """FastAPI dependency enforcing API key authentication.

    Behaviour:
        - Public paths are always allowed.
        - When no real secret is configured: in development the request is
          allowed with a logged warning; in production it is rejected.
        - When a secret is configured: a matching key is required.

    Args:
        request: The incoming request.

    Returns:
        The authenticated client's API key, or ``None`` in permissive dev mode.

    Raises:
        AuthenticationError: If authentication is required but missing/invalid.
    """
    settings = get_settings()
    if is_public_path(request.url.path):
        return None

    provided = get_api_key_from_headers(request.headers)

    if not _secret_is_configured(settings):
        if settings.is_production:
            logger.error("API_SECRET_KEY is unset/placeholder in production; rejecting request")
            raise AuthenticationError("Server API key is not configured. Set API_SECRET_KEY.")
        logger.warning(
            "API_SECRET_KEY is unset/placeholder; allowing request in development mode. "
            "Set a strong API_SECRET_KEY and ENVIRONMENT=production before deploying."
        )
        return provided

    if not provided:
        raise AuthenticationError("Missing API key. Supply X-API-Key or Authorization: Bearer.")
    if not verify_api_key(provided, settings.api_secret_key):
        raise AuthenticationError("Invalid API key.")
    return provided
