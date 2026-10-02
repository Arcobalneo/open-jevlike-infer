"""Command line: ``jevlike-infer serve`` and ``jevlike-infer models``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from jevlike_infer import __version__
from jevlike_infer.api.schema import Limits
from jevlike_infer.config import Settings
from jevlike_infer.media import MediaConfig
from jevlike_infer.models import MODELS, model_module


def _env(name: str, default):
    value = os.environ.get(f"JEVLIKE_{name}")
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes")
    return type(default)(value) if default is not None else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jevlike-infer", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="list supported models and backends")

    serve = sub.add_parser("serve", help="serve POST /v1/systemone")
    serve.add_argument("--model", default=_env("MODEL", "clef-flash"), choices=sorted(MODELS))
    serve.add_argument(
        "--model-path",
        default=_env("MODEL_PATH", None),
        required=_env("MODEL_PATH", None) is None,
        help="local directory with the model release (see scripts/download_model.py)",
    )
    serve.add_argument("--backend", default=_env("BACKEND", None), help="default: the model's first backend")
    serve.add_argument("--served-model-name", default=_env("SERVED_MODEL_NAME", ""))
    serve.add_argument("--host", default=_env("HOST", "0.0.0.0"))
    serve.add_argument("--port", type=int, default=_env("PORT", 8000))
    serve.add_argument("--api-key", default=_env("API_KEY", None), help="require 'Authorization: Bearer <key>'")
    serve.add_argument("--max-length", type=int, default=_env("MAX_LENGTH", 16384))
    serve.add_argument("--gpu-memory-utilization", type=float, default=_env("GPU_MEMORY_UTILIZATION", 0.5))
    serve.add_argument("--max-batch", type=int, default=_env("MAX_BATCH", 32))
    serve.add_argument("--batch-window-ms", type=float, default=_env("BATCH_WINDOW_MS", 2.0))
    serve.add_argument("--no-warmup", action="store_true", default=_env("NO_WARMUP", False))
    serve.add_argument("--max-media-items", type=int, default=_env("MAX_MEDIA_ITEMS", 16))
    serve.add_argument("--max-media-bytes", type=int, default=_env("MAX_MEDIA_BYTES", 50 * 1024 * 1024))
    serve.add_argument(
        "--no-media-urls",
        action="store_true",
        default=_env("NO_MEDIA_URLS", False),
        help="reject http(s) media references (only data URIs / base64)",
    )
    serve.add_argument("--video-frames", type=int, default=_env("VIDEO_FRAMES", 16))
    serve.add_argument("--frame-list-fps", type=float, default=_env("FRAME_LIST_FPS", 2.0))
    serve.add_argument("--log-level", default=_env("LOG_LEVEL", "info"))
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    backends = model_module(args.model).BACKENDS
    backend = args.backend or backends[0]
    if backend not in backends:
        raise SystemExit(f"model {args.model} supports backends {', '.join(backends)}, not {backend!r}")
    return Settings(
        model=args.model,
        model_path=args.model_path,
        backend=backend,
        served_model_name=args.served_model_name,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        max_length=args.max_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_batch=args.max_batch,
        batch_window_ms=args.batch_window_ms,
        warmup=not args.no_warmup,
        limits=Limits(max_media_items=args.max_media_items),
        media=MediaConfig(
            max_bytes=args.max_media_bytes,
            allow_urls=not args.no_media_urls,
            video_frames=args.video_frames,
            frame_list_fps=args.frame_list_fps,
        ),
    )


def serve(settings: Settings, log_level: str) -> None:
    import uvicorn

    from jevlike_infer.api.server import create_app
    from jevlike_infer.models import load_model

    log = logging.getLogger("jevlike_infer")
    started = time.time()
    log.info("loading %s (%s backend) from %s", settings.model, settings.backend, settings.model_path)
    model = load_model(settings)
    log.info("loaded in %.1fs", time.time() - started)
    if settings.warmup:
        started = time.time()
        model.warmup()
        log.info("warmed up in %.1fs", time.time() - started)
    uvicorn.run(create_app(model, settings), host=settings.host, port=settings.port, log_level=log_level)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "models":
        for name in sorted(MODELS):
            print(f"{name}\tbackends: {', '.join(model_module(name).BACKENDS)}")
        return
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    serve(settings_from_args(args), args.log_level)


if __name__ == "__main__":
    sys.exit(main())
