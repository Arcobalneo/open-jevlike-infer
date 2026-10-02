"""Clef-Flash on vLLM: the Qwen3.5 backbone runs as a vLLM pooling model (last hidden state of every
token), the release's joint schema head runs on top in the same process, and concurrent requests are
micro-batched into one vLLM call.

The pooling-model approach comes from ``clef_vllm.py`` in simonlehmann/clef-NVFP4 (Apache-2.0).
Changes: video placeholders are collapsed correctly (see ``tokens.py``), the token sequence vLLM
actually ran is checked against Clef's encoding, and requests are batched across callers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from jevlike_infer.batching import MicroBatcher
from jevlike_infer.config import Settings
from jevlike_infer.errors import RequestError
from jevlike_infer.models.base import DecisionModel
from jevlike_infer.models.clef_flash.release import import_release, to_release_record
from jevlike_infer.models.clef_flash.tokens import VisionTokens, collapse_media_placeholders


class ClefFlashVLLM(DecisionModel):
    def __init__(self, settings: Settings) -> None:
        # The engine runs in this process: no IPC hop per batch.
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoProcessor
        from vllm import LLM
        from vllm.config import PoolerConfig

        path = Path(settings.model_path)
        self.release = import_release(path)
        self.max_length = settings.max_length
        self.processor = AutoProcessor.from_pretrained(path)
        self.tokenizer = self.processor.tokenizer
        token_id = self.tokenizer.convert_tokens_to_ids
        self.vision_tokens = VisionTokens(
            image_pad=token_id("<|image_pad|>"),
            video_pad=token_id("<|video_pad|>"),
            vision_start=token_id("<|vision_start|>"),
            vision_end=token_id("<|vision_end|>"),
        )
        media_limit = settings.limits.max_media_items
        self.llm = LLM(
            model=str(path),
            runner="pooling",
            pooler_config=PoolerConfig(task="token_embed", tok_pooling_type="ALL", use_activation=False),
            max_model_len=settings.max_length,
            gpu_memory_utilization=settings.gpu_memory_utilization,
            limit_mm_per_prompt={"image": media_limit, "video": media_limit},
            enable_prefix_caching=False,
        )
        self.torch = torch
        self.device = torch.device("cuda")
        head = self.release.JointSchemaHead(**json.loads((path / "joint_head_config.json").read_text()))
        head.load_state_dict(load_file(path / "joint_head.safetensors"), strict=True)
        self.head = head.to(self.device, torch.bfloat16).eval()
        # The head reads lm_head rows as option embeddings; vLLM's pooling model does not load lm_head.
        weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        with safe_open(str(path / weight_map["lm_head.weight"]), framework="pt", device="cuda") as f:
            self.lm_head = f.get_tensor("lm_head.weight").to(torch.bfloat16)
        self.batcher: MicroBatcher[dict[str, Any], dict[str, Any]] = MicroBatcher(
            self._run_batch, max_batch=settings.max_batch, window=settings.batch_window_ms / 1000
        )

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.batcher.submit(request)

    def warmup(self) -> None:
        from jevlike_infer.models.clef_flash import warmup

        warmup(self, self.max_length)

    def _prompt(self, request: dict[str, Any], input_ids: tuple[int, ...]) -> dict[str, Any]:
        prompt: dict[str, Any] = {"prompt_token_ids": collapse_media_placeholders(input_ids, self.vision_tokens)}
        media: dict[str, Any] = {}
        images = request.get("images") or []
        videos = request.get("videos") or []
        if images:
            media["image"] = images if len(images) > 1 else images[0]
        if videos:
            # vLLM's Qwen3-VL processor builds timestamp tokens from per-video metadata; pass the same
            # metadata the HF processor got so both expansions agree.
            items = [(v.frames, {**v.metadata, "video_backend": "pyav", "do_sample_frames": False}) for v in videos]
            media["video"] = items if len(items) > 1 else items[0]
        if media:
            prompt["multi_modal_data"] = media
        if request.get("media_kwargs"):
            prompt["mm_processor_kwargs"] = dict(request["media_kwargs"])
        return prompt

    def _run_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any] | Exception]:
        torch = self.torch
        results: list[dict[str, Any] | Exception] = [RuntimeError("not run")] * len(requests)
        live: list[tuple[int, Any]] = []
        for index, request in enumerate(requests):
            try:
                record = to_release_record(request)
                encoded = self.release.encode_record(
                    self.tokenizer, record, max_length=self.max_length, processor=self.processor
                )
                live.append((index, encoded))
            except Exception as exc:  # a malformed request fails alone, not the batch
                results[index] = RequestError(str(exc)) if isinstance(exc, ValueError) else exc
        if not live:
            return results
        try:
            outputs = self._encode(requests, live)
        except Exception:
            if len(live) == 1:
                raise
            # One bad item (e.g. media vLLM rejects) must not fail its batch-mates: retry one by one.
            outputs = []
            for item in live:
                try:
                    outputs.extend(self._encode(requests, [item]))
                except Exception as exc:
                    outputs.append(RequestError(str(exc)) if isinstance(exc, ValueError) else exc)
        with torch.inference_mode():
            for (index, encoded), output in zip(live, outputs, strict=True):
                results[index] = (
                    output if isinstance(output, Exception) else self._answer(requests[index], encoded, output)
                )
        return results

    def _encode(self, requests: list[dict[str, Any]], items: list[tuple[int, Any]]) -> list[Any]:
        return self.llm.encode(
            [self._prompt(requests[index], encoded.input_ids) for index, encoded in items],
            pooling_task="token_embed",
            use_tqdm=False,
        )

    def _answer(self, request: dict[str, Any], encoded: Any, output: Any) -> dict[str, Any] | Exception:
        torch = self.torch
        hidden = output.outputs.data.to(self.device, torch.bfloat16)
        # The head reads hidden states at Clef's token positions, so vLLM must have run exactly Clef's
        # token sequence, not just one of the same length.
        if tuple(output.prompt_token_ids) != encoded.input_ids or hidden.shape[0] != len(encoded.input_ids):
            return RuntimeError(
                f"vLLM prompt ({len(output.prompt_token_ids)} tokens, {hidden.shape[0]} positions) "
                f"does not match Clef's encoding ({len(encoded.input_ids)} tokens)"
            )
        batch = self.release.collate_records([encoded], self.tokenizer.pad_token_id, self.device)
        logits = self.head(
            hidden.unsqueeze(0), batch["input_ids"], batch["attention_mask"], batch["records"], self.lm_head
        )[0]
        questions = request["questions"]
        return {
            "model": request["model"],
            "answers": {
                question.question_id: self.release.systemone_answer(
                    questions[question.question_id],
                    dict(zip(question.option_ids, question_logits.float().softmax(-1).tolist(), strict=True)),
                )
                for question, question_logits in zip(encoded.questions, logits, strict=True)
            },
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }
