"""First-request latency at input lengths the server has not seen yet.

    python benchmarks/cold_start.py --base-url http://127.0.0.1:8000

For each length, sends the same request twice and once more with one extra token. A first request
much slower than the second means a kernel was compiled or autotuned on the request path; start the
server without ``--no-warmup`` to pay that before traffic arrives.
"""

from __future__ import annotations

import argparse
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--lengths", default="61,263,1213,2501,5003,7019,8101,9551,11777,13007,15013,16500")
    args = parser.parse_args()
    client = httpx.Client(timeout=600)
    url = f"{args.base_url.rstrip('/')}/v1/systemone"

    def hit(words: int, word: str) -> tuple[int, float]:
        body = {"model": "cold", "state": f"{word} " * words, "questions": {"q": {"type": "noul"}}}
        started = time.perf_counter()
        response = client.post(url, json=body)
        response.raise_for_status()
        return response.json()["usage"]["input_tokens"], (time.perf_counter() - started) * 1000

    for words in (int(x) for x in args.lengths.split(",")):
        tokens, first = hit(words, "gamma")
        _, second = hit(words, "gamma")
        _, plus_one = hit(words + 1, "gamma")
        print(
            f"tokens={tokens:6d}  first={first:8.0f} ms  second={second:7.0f} ms  +1 token={plus_one:7.0f} ms",
            flush=True,
        )


if __name__ == "__main__":
    main()
