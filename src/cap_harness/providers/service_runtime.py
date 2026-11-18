# Modified by FF EAI Robotics Ecosystem Inc. (2026).

from __future__ import annotations

import asyncio
import os

_UNBOUNDED_PATHS = frozenset({"/openapi.json", "/docs", "/redoc", "/healthz"})


def request_limit(default: int = 1) -> int:
    raw = os.environ.get("CAP_HARNESS_SERVICE_CONCURRENCY")
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            f"CAP_HARNESS_SERVICE_CONCURRENCY must be an integer, got {raw!r}"
        ) from error
    if value < 1:
        raise ValueError("CAP_HARNESS_SERVICE_CONCURRENCY must be >= 1")
    return value


def install_service_runtime(app, *, limit: int | None = None):
    resolved = request_limit() if limit is None else limit
    semaphore = asyncio.Semaphore(resolved)
    app.state.request_limit = resolved

    @app.middleware("http")
    async def _bounded(request, call_next):
        if request.url.path in _UNBOUNDED_PATHS:
            return await call_next(request)
        async with semaphore:
            return await call_next(request)

    @app.get("/healthz")
    async def _healthz():
        return {"ok": True, "request_limit": app.state.request_limit}

    return app
