# Adding a model

1. Create `src/jevlike_infer/models/<name>/__init__.py` with:

   ```python
   BACKENDS = ("vllm",)  # first one is the default


   def load(settings: Settings) -> DecisionModel: ...
   ```

2. Implement `DecisionModel` (`models/base.py`):
   - `decide(request) -> dict` receives a validated request with decoded media and returns the
     SystemOne response body. It is called from many threads at once; use
     `jevlike_infer.batching.MicroBatcher` to batch, or a lock to serialize.
   - Raise `RequestError` (or `ValueError`) for inputs the model cannot take; they become HTTP 400.
   - `warmup()` (optional) runs before the server accepts traffic.
   - Import heavy dependencies (torch, vllm) inside the backend module, not at package import, so
     the registry and unit tests stay light.
3. Register it in `MODELS` in `src/jevlike_infer/models/__init__.py` and in `scripts/download_model.py`
   with a pinned revision.
4. Add `docs/models/<name>.md` (model facts, backends, validated hardware, benchmark results) and a
   row to the support table in both READMEs.
5. Run `tests/e2e/run_e2e.py` against it and `benchmarks/compare_servers.py` against the model's
   reference implementation. Post the numbers in the pull request.

Model-specific behavior (prompt format, placeholder handling, processor quirks) stays inside the
model package. If the shared layers need to change for a new model, open an issue first.
