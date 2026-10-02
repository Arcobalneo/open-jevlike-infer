"""The interface every supported model implements.

The HTTP layer validates the request and decodes media, then hands the model a *decoded request*: the
original body with ``images`` as a list of RGB ``PIL.Image`` and ``videos`` as a list of
:class:`jevlike_infer.media.Video`. The model returns a SystemOne response body::

    {"model": ..., "answers": {question_id: answer, ...}, "usage": {"input_tokens": n, "output_tokens": 0}}

where an answer is ``{"type": "choice", "choice", "confidence", "probabilities"}``,
``{"type": "score", "score", "confidence", "legend", "probabilities"}`` or ``{"type": "noul", "noul"}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DecisionModel(ABC):
    @abstractmethod
    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        """Answer one decoded request. Must be safe to call from many threads at once.

        Raise :class:`jevlike_infer.errors.RequestError` (or ``ValueError``) for inputs the model cannot
        accept, such as a schema longer than the context window.
        """

    def warmup(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Pay one-time costs (kernel compilation, autotuning) before serving traffic."""
