# Clef-Flash

| | |
|---|---|
| Model | [Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash), revision `17f0b0a` |
| Architecture | Qwen3.5-9B (with vision encoder) + joint schema head, bf16, ~19 GB |
| License | Apache-2.0 |
| Context | 16384 tokens (text, schema and media together) |
| Inputs | Text or JSON state, images, videos |
| Backends | `vllm` (default), `hf` |
| Validated on | NVIDIA A800-SXM4-80GB, driver 550 (CUDA 12.4) |

## Run

```bash
python scripts/download_model.py --model clef-flash --dir ./models/clef-flash
jevlike-infer serve --model clef-flash --model-path ./models/clef-flash --port 8000
```

GPU memory: the vLLM backend reserves `--gpu-memory-utilization` of the card (default 0.5, about 40
GB on an 80 GB card) for weights, activations and cache. Weights alone are ~19 GB; lower the fraction
on smaller cards, keeping room for 16k-token inputs. Start-up takes ~100 s on an A800 (loading ~88 s,
warm-up ~12 s).

## Backends

| | `vllm` | `hf` |
|---|---|---|
| Runtime | vLLM 0.30 pooling model + joint head | transformers, the release's `systemone()` |
| Concurrency | dynamic micro-batching | one request at a time |
| Extra install | `scripts/install_cuda12.sh` on drivers < 580 | `pip install -e ".[hf]"` plus `causal-conv1d` |
| Use | production | numerical reference, fallback |

Both backends give the same decisions. On the comparison set in `benchmarks/compare_servers.py`
the top option matches on every question and the largest total-variation distance between the two
distributions is below 0.01.

## Results

See [benchmarks/results/clef-flash-a800.md](../../benchmarks/results/clef-flash-a800.md) for full
numbers and conditions.

## Known limits

- The release's joint head scores one record at a time; it is the largest per-request cost after the
  backbone at high concurrency.
- The Jev API does not specify how media travel over HTTP. This server accepts data URIs, base64,
  URLs and frame lists (see [API reference](../api.md)); a future official client may differ.
- `media_kwargs` changes how images are tokenized. On the vLLM backend the result is checked token by
  token; unsupported combinations fail with a 500 error naming the mismatch.
