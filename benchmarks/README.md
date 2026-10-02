# Benchmarks

All scripts talk to a running server unless noted, so they measure what clients see.

| Script | Measures |
|---|---|
| `bench_latency.py` | Serial latency, throughput at several concurrency levels, latency by input length, image latency |
| `bench_arc.py` | ARC-Challenge / ARC-Easy accuracy through the API (`pip install -e ".[bench]"`) |
| `compare_servers.py` | Per-question agreement between two servers, e.g. `vllm` vs `hf` backend |
| `cold_start.py` | First-request latency at unseen input lengths |
| `profile_clef_flash.py` | Encoding / backbone / head time per batch; loads its own engine, needs a free GPU |

Published results:

- [Clef-Flash on A800](results/clef-flash-a800.md)

To add results for new hardware, run the scripts with `--output`, put the files in
`results/<model>-<gpu>/`, write a `results/<model>-<gpu>.md` with the same sections, and open a pull
request.
