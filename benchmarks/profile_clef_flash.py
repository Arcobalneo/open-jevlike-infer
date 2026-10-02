"""Time the stages of one Clef-Flash vLLM batch: CPU encoding, vLLM backbone, joint head.

    CUDA_VISIBLE_DEVICES=0 python benchmarks/profile_clef_flash.py --model-path ./models/clef-flash

Loads its own engine (needs a free GPU); does not talk to a running server.
"""

from __future__ import annotations

import argparse
import time

from jevlike_infer.config import Settings
from jevlike_infer.models.clef_flash.release import to_release_record
from jevlike_infer.models.clef_flash.vllm_backend import ClefFlashVLLM

REQUEST = {
    "model": "profile",
    "state": "Our checkout started returning errors and orders are blocked.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": "Bugs"},
        },
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
        "outage": {"type": "noul", "instructions": "Is a service down?"},
    },
    "images": [],
    "videos": [],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-sizes", default="1,1,4,16,32")
    args = parser.parse_args()
    model = ClefFlashVLLM(Settings(model_path=args.model_path))
    torch = model.torch
    for n in (int(x) for x in args.batch_sizes.split(",")):
        requests = [dict(REQUEST, state=f"{REQUEST['state']} #{i}") for i in range(n)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        encoded = [
            model.release.encode_record(
                model.tokenizer, to_release_record(r), max_length=model.max_length, processor=model.processor
            )
            for r in requests
        ]
        t1 = time.perf_counter()
        outputs = model._encode(requests, list(enumerate(encoded)))
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        with torch.inference_mode():
            for request, enc, output in zip(requests, encoded, outputs, strict=True):
                model._answer(request, enc, output)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        print(
            f"batch {n:3d}: encode {1000 * (t1 - t0):7.1f} ms  backbone {1000 * (t2 - t1):7.1f} ms  "
            f"head {1000 * (t3 - t2):7.1f} ms  per request {1000 * (t3 - t0) / n:6.1f} ms"
        )


if __name__ == "__main__":
    main()
