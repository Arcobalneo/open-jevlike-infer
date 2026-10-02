# Clef-Flash on A800: results

Measured 2026-10-02 with open-jevlike-infer 0.1.0. Raw outputs are in [`clef-flash-a800/`](clef-flash-a800/).

## Setup

| | |
|---|---|
| GPU | 1x NVIDIA A800-SXM4-80GB (one card per server), driver 550.163.01 (CUDA 12.4) |
| CPU | Intel Xeon Platinum 8350C |
| Host | Shared machine; other GPUs ran training jobs during the measurements |
| Model | Cloudflare/clef-flash `17f0b0a`, bf16 |
| `vllm` backend | vLLM 0.30.0+cu129, torch 2.13.0+cu129, transformers 5.18.0, Python 3.10; default settings (`--gpu-memory-utilization 0.5`, `--max-batch 32`) |
| `hf` backend | torch 2.11.0+cu126, transformers 5.10.2, flash-linear-attention 0.5.2, causal-conv1d 1.7.0 |
| Client | Same host, HTTP over loopback |

## Accuracy

`benchmarks/bench_arc.py`, full test splits, through the HTTP API.

| Benchmark | Model card | `vllm` backend |
|---|---|---|
| ARC-Challenge (1172) | 98.3 | **98.6** |
| ARC-Easy (2376) | 99.5 | **99.6** |

## Backend agreement

`benchmarks/compare_servers.py`, `hf` vs `vllm`: the top option matches on all 9 questions (text,
JSON, Chinese, 11k-token, two-image and video requests); the largest total-variation distance is
0.0067. Both backends pass all 87 checks of `tests/e2e/run_e2e.py`.

## Latency and throughput

`benchmarks/bench_latency.py`. The short request is 3 questions (choice, score, noul), 291 tokens.

| | `vllm` | `hf` |
|---|---|---|
| Short request, serial, p50 | **45 ms** | 60 ms |
| Short request, serial, p95 | 46 ms | 119 ms |
| Concurrency 16, throughput | **28 req/s** | 15 req/s |
| Concurrency 32, throughput | **33 req/s** | - |
| 512x512 image + 3 questions (550 tokens), p50 | **75 ms** | 177 ms |

Latency by input length (p50):

| Input tokens | `vllm` | `hf` |
|---|---|---|
| 447 | 58 ms | 63 ms |
| 942 | 96 ms | 95 ms |
| 2922 | 231 ms | 271 ms |
| 5562 | 433 ms | 441 ms |
| 10842 | 839 ms | 854 ms |
| 16122 | 1360 ms | 1271 ms |

Short and image requests gain the most from vLLM. Above ~1k tokens both backends are bound by the
backbone's prefill and perform about the same. Cloudflare reports a 38.8 ms median on an H200.

## Cold start

`benchmarks/cold_start.py`, first request at lengths not seen before. With warm-up (default) no
length stalls on either backend. Without warm-up the `vllm` backend pays ~0.3 s once on the very
first request; the `hf` backend pays ~12 s on the first request of every new 1024-token bucket.

## Where the time goes

`benchmarks/profile_clef_flash.py`, batches of the short request, `vllm` backend:

| Batch | Encoding | Backbone | Joint head | Per request |
|---|---|---|---|---|
| 1 | 2.2 ms | 34.8 ms | 7.1 ms | 44.1 ms |
| 4 | 7.5 ms | 90.7 ms | 26.3 ms | 31.1 ms |
| 16 | 28.7 ms | 315.3 ms | 102.3 ms | 27.9 ms |
| 32 | 56.8 ms | 604.1 ms | 218.5 ms | 27.5 ms |
