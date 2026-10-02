"""Clef-Flash on transformers: the release's own ``load_release_model`` and ``systemone``, one request at
a time under a GPU lock. Slower than the vLLM backend; kept as the numerical reference and as a
fallback for environments without vLLM.

Install ``flash-linear-attention`` and ``causal-conv1d`` (the ``hf`` extra): without them the
Qwen3.5 linear-attention layers fall back to plain torch and short requests get ~2.5x slower.
"""

from __future__ import annotations

import threading
from typing import Any

from jevlike_infer.config import Settings
from jevlike_infer.errors import OverloadedError
from jevlike_infer.models.base import DecisionModel
from jevlike_infer.models.clef_flash.release import import_release, to_release_record


class ClefFlashHF(DecisionModel):
    def __init__(self, settings: Settings) -> None:
        import torch

        self.torch = torch
        self.release = import_release(settings.model_path)
        self.max_length = settings.max_length
        self.model, self.processor = self.release.load_release_model(settings.model_path, device="cuda")
        self.lock = threading.Lock()

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        record = to_release_record(request)
        with self.lock:
            try:
                return self.release.systemone(self.model, self.processor, record, max_length=self.max_length)
            except self.torch.cuda.OutOfMemoryError as exc:
                self.torch.cuda.empty_cache()
                raise OverloadedError("GPU out of memory; retry with a smaller input") from exc

    def warmup(self) -> None:
        from jevlike_infer.models.clef_flash import warmup

        warmup(self, self.max_length)
