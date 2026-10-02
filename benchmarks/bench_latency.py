"""Latency and throughput of a running server.

    python benchmarks/bench_latency.py --base-url http://127.0.0.1:8000 [--output results.json]

Measures, in order: serial latency of a short 3-question request, throughput and latency at several
concurrency levels, latency by input length, and latency of a 512x512 image request.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import io
import json
import statistics
import time

import httpx
from PIL import Image

SHORT = {
    "model": "bench",
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
}


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(q * len(ordered)) - 1))]


def timed(client: httpx.Client, url: str, body: dict) -> tuple[float, dict]:
    started = time.perf_counter()
    response = client.post(url, json=body)
    response.raise_for_status()
    return (time.perf_counter() - started) * 1000, response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--serial", type=int, default=100)
    parser.add_argument("--concurrency", default="1,4,16,32,64")
    parser.add_argument("--requests-per-level", type=int, default=256)
    parser.add_argument("--output")
    args = parser.parse_args()
    url = f"{args.base_url.rstrip('/')}/v1/systemone"
    client = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=256))
    report: dict = {}

    for _ in range(5):
        timed(client, url, SHORT)
    serial = [timed(client, url, SHORT)[0] for _ in range(args.serial)]
    tokens = timed(client, url, SHORT)[1]["usage"]["input_tokens"]
    report["serial"] = {"input_tokens": tokens, "p50_ms": statistics.median(serial), "p95_ms": percentile(serial, 0.95)}
    print(f"serial ({tokens} tokens): p50 {report['serial']['p50_ms']:.1f} ms, p95 {report['serial']['p95_ms']:.1f} ms")

    report["concurrency"] = []
    for level in (int(x) for x in args.concurrency.split(",")):
        started = time.perf_counter()
        with cf.ThreadPoolExecutor(level) as pool:
            latencies = [ms for ms, _ in pool.map(lambda _: timed(client, url, SHORT), range(args.requests_per_level))]
        wall = time.perf_counter() - started
        row = {
            "concurrency": level,
            "requests_per_s": args.requests_per_level / wall,
            "p50_ms": statistics.median(latencies),
            "p95_ms": percentile(latencies, 0.95),
        }
        report["concurrency"].append(row)
        print(
            f"concurrency {level:3d}: {row['requests_per_s']:6.1f} req/s, "
            f"p50 {row['p50_ms']:7.1f} ms, p95 {row['p95_ms']:7.1f} ms"
        )

    report["length"] = []
    for repeat in (15, 60, 240, 480, 960, 1440):
        body = dict(SHORT, state="The service status log shows normal operation and stable latency. " * repeat)
        timed(client, url, body)
        runs = [timed(client, url, body) for _ in range(5)]
        row = {"input_tokens": runs[0][1]["usage"]["input_tokens"], "p50_ms": statistics.median(ms for ms, _ in runs)}
        report["length"].append(row)
        print(f"length {row['input_tokens']:6d} tokens: p50 {row['p50_ms']:7.1f} ms")

    buffer = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(buffer, format="PNG")
    image_body = dict(SHORT, images=["data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()])
    timed(client, url, image_body)
    runs = [timed(client, url, image_body) for _ in range(20)]
    report["image_512"] = {
        "input_tokens": runs[0][1]["usage"]["input_tokens"],
        "p50_ms": statistics.median(ms for ms, _ in runs),
    }
    print(f"512x512 image ({report['image_512']['input_tokens']} tokens): p50 {report['image_512']['p50_ms']:.1f} ms")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
