"""Dynamic micro-batching: concurrent callers share one model call.

A background thread takes the first queued item, waits up to ``window`` seconds for more, drains what
already arrived (up to ``max_batch``) and runs them in one ``run_batch`` call. ``run_batch`` returns
one result per item; an ``Exception`` in that list fails only its own caller.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future
from typing import Generic, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class MicroBatcher(Generic[T, R]):
    def __init__(
        self,
        run_batch: Callable[[list[T]], list[R | Exception]],
        max_batch: int = 32,
        window: float = 0.002,
        name: str = "micro-batcher",
    ) -> None:
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self._run_batch = run_batch
        self._max_batch = max_batch
        self._window = window
        self._queue: queue.Queue[tuple[T, Future]] = queue.Queue()
        threading.Thread(target=self._loop, daemon=True, name=name).start()

    def submit(self, item: T) -> R:
        """Queue ``item`` and block until the batch containing it has run."""
        future: Future = Future()
        self._queue.put((item, future))
        return future.result()

    def _collect(self) -> list[tuple[T, Future]]:
        items = [self._queue.get()]
        timeout = self._window
        while len(items) < self._max_batch:
            try:
                items.append(self._queue.get(timeout=timeout))
            except queue.Empty:
                break
            timeout = 0  # drain what already arrived; do not keep waiting
        return items

    def _loop(self) -> None:
        while True:
            items = self._collect()
            try:
                results = self._run_batch([item for item, _ in items])
                if len(results) != len(items):
                    raise RuntimeError(f"run_batch returned {len(results)} results for {len(items)} items")
            except Exception as exc:  # the whole batch failed
                results = [exc] * len(items)
            for (_, future), result in zip(items, results, strict=True):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
