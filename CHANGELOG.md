# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-02

First release.

### Added

- `POST /v1/systemone` server compatible with the Jev / SystemOne API: `choice`, `score` and `noul`
  questions, text / JSON state, images and videos, Jev request limits, one error format.
- Cloudflare Clef-Flash support with two backends:
  - `vllm` (default): backbone as a vLLM 0.30 pooling model, joint schema head in process,
    dynamic micro-batching across concurrent requests.
  - `hf`: the release's transformers implementation, as the numerical reference and a fallback.
- Start-up warm-up across all input-length buckets.
- `scripts/install_cuda12.sh` for NVIDIA drivers older than 580; Dockerfile and systemd unit.
- `scripts/download_model.py` with pinned revision and sha256 verification.
- Unit tests (no GPU), end-to-end suite (87 checks), benchmarks for latency, throughput, ARC accuracy,
  cold start and backend comparison.

### Fixed (relative to the reference code this builds on)

- Videos sent to Qwen3-VL processors without metadata were resampled to 2 frames; the server now
  samples frames itself and passes metadata.
- Video placeholders were collapsed incorrectly before vLLM re-expanded them, which broke every video
  request on the vLLM path.
- vLLM's expansion is now checked token by token against Clef's encoding, not by length only.

### Deployment notes

- The Docker image installs `gcc`: Triton compiles kernel launchers when the server starts.
- `scripts/install_cuda12.sh` retries interrupted wheel downloads and accepts `VLLM_WHEEL_URL` for a mirror.
- `tests/e2e/run_e2e.py --fixture-host` lets a containerized server fetch the suite's test media.

[0.1.0]: https://github.com/Arcobalneo/open-jevlike-infer/releases/tag/v0.1.0
