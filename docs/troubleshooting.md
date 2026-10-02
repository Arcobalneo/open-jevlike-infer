# Troubleshooting

Problems hit while deploying Clef-Flash, with the fix for each.

### `The NVIDIA driver on your system is too old (found version 12040)`

PyPI's `vllm==0.30.0` and `torch==2.13.0` are CUDA 13 builds and need driver 580 or newer. On a
CUDA 12 driver, install with `scripts/install_cuda12.sh`, which uses the `+cu129` builds.

### `OSError: Could not load this library: .../torchcodec/libtorchcodec_image.so` / `libnvrtc.so.13`

vLLM imports torchcodec, and PyPI's torchcodec is a CUDA 13 build. Install
`torchcodec==0.16.0+cu129` from `https://download.pytorch.org/whl/cu129` (the CUDA 12 script does).

### Package installation hangs

When one `pip` / `uv` resolve uses both the PyTorch index and PyPI, generic packages (numpy,
networkx, ...) may be fetched from the PyTorch CDN. If that CDN is slow or unreachable from your
network, the install stalls. Install the torch packages first with `--no-deps` from the PyTorch
index, then the rest from PyPI, as `scripts/install_cuda12.sh` does.

### Short requests take ~160 ms on the transformers backend

`flash-linear-attention` and `causal-conv1d` are missing, so Qwen3.5's linear-attention layers run in
plain torch. The log shows `The fast path is not available`. Install both (the `hf` extra installs
the first; `causal-conv1d` needs `nvcc` and `pip install --no-build-isolation causal-conv1d`). With
them, short requests take ~62 ms. The vLLM backend has its own kernels and does not need them.

### The first request at a new input length takes 12 s (transformers backend)

On the `hf` backend, flash-linear-attention autotunes its kernels per input-length bucket. The
server sweeps every bucket at start-up; do not pass `--no-warmup` with `--backend hf`.
`benchmarks/cold_start.py` shows whether any length is still cold. The `vllm` backend does not have
this stall.

### A video answer ignores most of the clip

Qwen3-VL processors assume 24 fps when they get no video metadata and resample to 2 fps, so a
16-frame clip becomes 2 frames. This server samples frames itself, passes the metadata and turns
the processor's sampling off. If you call the model code directly, do the same
(`do_sample_frames=False`, `video_metadata=[...]`).

### `vLLM prompt (...) does not match Clef's encoding`

vLLM expanded the media placeholders differently from the HF processor, so hidden states would not
line up with Clef's question and option spans. The request fails instead of returning wrong answers.
Check that the transformers version in the vLLM environment matches the one vLLM was tested with,
and open an issue with the request.

### Hugging Face downloads stall behind a proxy

`huggingface_hub`'s Xet transfer can stall behind some HTTP proxies. Set `HF_HUB_DISABLE_XET=1`, or
use a mirror with `HF_ENDPOINT`. `scripts/download_model.py` resumes when re-run.
