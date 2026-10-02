"""Cloudflare Clef-Flash: Qwen3.5-9B multimodal backbone plus a joint schema head that scores every
option of every question in one prefill pass. https://huggingface.co/Cloudflare/clef-flash
"""

from __future__ import annotations

from jevlike_infer.config import Settings
from jevlike_infer.models.base import DecisionModel

BACKENDS = ("vllm", "hf")


def load(settings: Settings) -> DecisionModel:
    if settings.backend == "vllm":
        from jevlike_infer.models.clef_flash.vllm_backend import ClefFlashVLLM

        return ClefFlashVLLM(settings)
    from jevlike_infer.models.clef_flash.hf_backend import ClefFlashHF

    return ClefFlashHF(settings)


def warmup(model: DecisionModel, max_length: int) -> None:
    """Run one image request and one text request per 1024-token bucket up to ``max_length``.

    On the ``hf`` backend, flash-linear-attention autotunes its kernels per input-length bucket and
    the first request in each new bucket takes ~12 s. On the ``vllm`` backend there is no such stall;
    warm-up only removes the ~0.3 s overhead of the very first request.
    """
    from PIL import Image

    questions = {"q": {"type": "noul"}, "c": {"type": "choice", "criteria": {"a": None, "b": None}}}
    model.decide(
        {
            "model": "warmup",
            "state": "warmup",
            "questions": questions,
            "images": [Image.new("RGB", (224, 224), "white")],
            "videos": [],
        }
    )
    for tokens in range(512, max_length + 1024, 1024):
        model.decide({"model": "warmup", "state": " a" * tokens, "questions": questions, "images": [], "videos": []})
