"""Server settings. Every field can be set by a CLI flag or a ``JEVLIKE_*`` environment variable."""

from __future__ import annotations

from dataclasses import dataclass, field

from jevlike_infer.api.schema import Limits
from jevlike_infer.media import MediaConfig


@dataclass
class Settings:
    model: str = "clef-flash"
    model_path: str = ""
    backend: str = "vllm"
    served_model_name: str = ""
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    max_length: int = 16384
    # vLLM backend
    gpu_memory_utilization: float = 0.5
    max_batch: int = 32
    batch_window_ms: float = 2.0
    warmup: bool = True
    limits: Limits = field(default_factory=Limits)
    media: MediaConfig = field(default_factory=MediaConfig)

    @property
    def model_name(self) -> str:
        return self.served_model_name or self.model
