"""FastAPI serving layer for FinSage-7B."""

from __future__ import annotations

__all__ = ["create_app"]


def __getattr__(name: str) -> object:
    """Lazily expose the app factory to avoid importing FastAPI eagerly."""
    if name == "create_app":
        from finsage.serving.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
