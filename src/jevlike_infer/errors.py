"""Errors that map to HTTP responses."""

from __future__ import annotations


class RequestError(ValueError):
    """The request is malformed or violates a limit. Reported as HTTP 400."""


class OverloadedError(RuntimeError):
    """The server cannot take the request right now (e.g. GPU out of memory). Reported as HTTP 503."""
