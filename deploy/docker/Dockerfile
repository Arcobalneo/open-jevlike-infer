# open-jevlike-infer with the vLLM backend, CUDA 12.9 build (runs on NVIDIA driver >= 525).
#
#   docker build -f deploy/docker/Dockerfile -t open-jevlike-infer .
#   docker run --gpus '"device=0"' -p 8000:8000 -v /path/to/models/clef-flash:/models/clef-flash:ro \
#       open-jevlike-infer
#
# Build args: PYPI_INDEX (optional mirror), GITHUB_PROXY (optional, to fetch the vLLM wheel).
FROM python:3.12-slim

ARG PYPI_INDEX=""
ARG GITHUB_PROXY=""
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY scripts/install_cuda12.sh ./scripts/
RUN PYTHON=python3 scripts/install_cuda12.sh /opt/venv && rm -rf .cache /root/.cache

ENV PATH=/opt/venv/bin:$PATH \
    JEVLIKE_MODEL=clef-flash \
    JEVLIKE_MODEL_PATH=/models/clef-flash \
    JEVLIKE_PORT=8000
EXPOSE 8000
HEALTHCHECK --start-period=300s --interval=30s CMD curl -fs http://127.0.0.1:8000/health || exit 1
ENTRYPOINT ["jevlike-infer", "serve"]
