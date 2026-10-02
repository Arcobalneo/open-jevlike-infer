"""FastAPI app serving ``POST /v1/systemone``, ``GET /health`` and ``GET /v1/models``."""

from __future__ import annotations

import hmac
import logging
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from jevlike_infer import __version__
from jevlike_infer.api.schema import validate_request
from jevlike_infer.config import Settings
from jevlike_infer.errors import OverloadedError, RequestError
from jevlike_infer.media import decode_image, decode_video
from jevlike_infer.models.base import DecisionModel

log = logging.getLogger("jevlike_infer")


def error_response(status: int, error_type: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": error_type, "message": message}})


def create_app(model: DecisionModel, settings: Settings) -> FastAPI:
    app = FastAPI(title="open-jevlike-infer", version=__version__)
    expected_auth = f"Bearer {settings.api_key}" if settings.api_key else None

    def decide(body: Any) -> JSONResponse:
        started = time.perf_counter()
        try:
            validate_request(body, settings.limits)
            request = dict(body)
            request["images"] = [decode_image(ref, settings.media) for ref in body.get("images") or []]
            request["videos"] = [decode_video(ref, settings.media) for ref in body.get("videos") or []]
            response = model.decide(request)
        except (RequestError, ValueError) as exc:
            return error_response(400, "invalid_request_error", str(exc))
        except OverloadedError as exc:
            return error_response(503, "overloaded_error", str(exc))
        except Exception as exc:
            log.exception("inference failed")
            return error_response(500, "api_error", f"{type(exc).__name__}: {exc}")
        log.info(
            "systemone questions=%d images=%d videos=%d input_tokens=%d latency_ms=%.1f",
            len(body["questions"]),
            len(request["images"]),
            len(request["videos"]),
            response["usage"]["input_tokens"],
            (time.perf_counter() - started) * 1000,
        )
        return JSONResponse(content=response)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": settings.model_name}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": settings.model_name, "object": "model"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        if expected_auth and not hmac.compare_digest(request.headers.get("authorization", ""), expected_auth):
            return error_response(401, "authentication_error", "invalid or missing bearer token")
        try:
            body = await request.json()
        except Exception:
            return error_response(400, "invalid_request_error", "request body must be valid JSON")
        return await run_in_threadpool(decide, body)

    return app
