"""ARC-Challenge and ARC-Easy accuracy through the HTTP API, to check a deployment against the model card.

    pip install -e ".[bench]"
    python benchmarks/bench_arc.py --base-url http://127.0.0.1:8000 [--limit 200]

Each test question is sent as one ``choice`` question whose options are the answer texts keyed by
their letters. Clef-Flash's model card reports ARC-Challenge 98.3 and ARC-Easy 99.5.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time

import httpx
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SPLITS = {
    "ARC-Challenge": "ARC-Challenge/test-00000-of-00001.parquet",
    "ARC-Easy": "ARC-Easy/test-00000-of-00001.parquet",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, help="first N questions per split")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output")
    args = parser.parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/systemone"
    client = httpx.Client(timeout=300)
    report = {}

    def answer(row: dict) -> bool:
        criteria = dict(zip(row["choices"]["label"], row["choices"]["text"], strict=True))
        body = {
            "model": "bench",
            "state": {"question": row["question"]},
            "questions": {
                "answer": {
                    "type": "choice",
                    "instructions": "Which option correctly answers the question?",
                    "criteria": criteria,
                }
            },
        }
        response = client.post(url, json=body)
        response.raise_for_status()
        return response.json()["answers"]["answer"]["choice"] == row["answerKey"]

    for name, filename in SPLITS.items():
        rows = pq.read_table(hf_hub_download("allenai/ai2_arc", filename, repo_type="dataset")).to_pylist()[
            : args.limit
        ]
        started = time.perf_counter()
        with cf.ThreadPoolExecutor(args.concurrency) as pool:
            correct = sum(pool.map(answer, rows))
        report[name] = {"n": len(rows), "accuracy": 100 * correct / len(rows), "seconds": time.perf_counter() - started}
        print(f"{name}: {report[name]['accuracy']:.1f}% of {len(rows)} ({report[name]['seconds']:.0f} s)")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
