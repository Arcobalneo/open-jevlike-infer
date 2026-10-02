"""Token-level helpers for running Clef's encoding through vLLM. Pure Python, no torch."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class VisionTokens:
    image_pad: int
    video_pad: int
    vision_start: int
    vision_end: int


def collapse_media_placeholders(input_ids: Sequence[int], tokens: VisionTokens) -> list[int]:
    """Undo the HF processor's placeholder expansion so vLLM can redo it from the media.

    Clef encodes a record with the HF processor, which expands every media placeholder. vLLM expects
    one placeholder per item and expands it again itself. Collapsing exactly what the processor
    expanded makes vLLM's hidden states line up with Clef's token positions:

    * image: ``<vision_start> <image_pad>*N <vision_end>`` keeps a single ``<image_pad>``;
    * Qwen3-VL video: ``<vision_start> (<t seconds> <vision_start> <video_pad>*N <vision_end>)*T
      <vision_end>`` becomes ``<vision_start> <video_pad> <vision_end>``.

    Collapsing only repeated pad tokens (enough for images) breaks videos, whose per-frame-group
    timestamps and nested vision markers would be duplicated by vLLM's own expansion.
    """
    ids: list[int] = []
    i, n = 0, len(input_ids)
    while i < n:
        token = input_ids[i]
        ids.append(token)
        i += 1
        if token == tokens.image_pad:
            while i < n and input_ids[i] == tokens.image_pad:
                i += 1
        elif token == tokens.vision_start and i < n and input_ids[i] != tokens.image_pad:
            depth = 1
            while i < n and depth:
                if input_ids[i] == tokens.vision_start:
                    depth += 1
                elif input_ids[i] == tokens.vision_end:
                    depth -= 1
                i += 1
            ids += [tokens.video_pad, tokens.vision_end]
    return ids
