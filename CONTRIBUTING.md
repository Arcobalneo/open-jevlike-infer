# Contributing

Thanks for helping. Issues and pull requests are welcome, especially:

- support for another open Jev-like decision model (see [docs/adding-a-model.md](docs/adding-a-model.md));
- benchmark results on hardware not listed yet (H100, H200, L40S, RTX 4090, ...);
- performance work on the serving path.

## Development setup

```bash
git clone https://github.com/Arcobalneo/open-jevlike-infer.git
cd open-jevlike-infer
pip install -e ".[dev]"
pytest            # unit tests, no GPU needed
ruff check . && ruff format --check .
```

GPU checks, against a running server:

```bash
python tests/e2e/run_e2e.py --base-url http://127.0.0.1:8000
python benchmarks/compare_servers.py --a http://127.0.0.1:8000 --b http://127.0.0.1:8001  # e.g. vllm vs hf
```

## Pull requests

- One topic per pull request. Keep model-specific code inside `src/jevlike_infer/models/<name>/`.
- Add or update unit tests for anything that can run without a GPU.
- Changes to the inference path need e2e results, and `compare_servers.py` output when they can
  change numbers. Paste them in the pull request with GPU, driver and package versions.
- Update the docs and `CHANGELOG.md` in the same pull request.

## Reporting a bug

Include the request body (or a minimal one that reproduces it), the response, the server log lines
around it, and the output of `jevlike-infer --version`, `nvidia-smi` and `pip list | grep -E "vllm|torch|transformers"`.
