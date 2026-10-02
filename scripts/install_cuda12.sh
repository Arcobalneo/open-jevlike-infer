#!/usr/bin/env bash
# Install open-jevlike-infer with the vLLM backend on a CUDA 12 driver (525 <= driver < 580).
#
# PyPI's vllm 0.30.0 and torch 2.13.0 are built for CUDA 13 and need driver >= 580. On older drivers
# this script installs the CUDA 12.9 builds instead, which run on any 12.x driver through CUDA minor
# version compatibility:
#   vllm 0.30.0+cu129 (GitHub release wheel), torch / torchvision / torchaudio / torchcodec +cu129.
# torch and friends are installed first from the PyTorch index with --no-deps; everything else then
# comes from your normal PyPI index (mixing both indexes in one resolve makes pip/uv fetch generic
# packages such as numpy from the PyTorch CDN, which is slow or unreachable in some networks).
#
# Usage:  scripts/install_cuda12.sh [VENV_DIR]        (default .venv; needs uv: pip install uv)
# Env:    PYTHON (default python3), PYPI_INDEX (optional mirror), GITHUB_PROXY (optional, for the wheel)
set -euo pipefail

VENV=${1:-.venv}
PYTHON=${PYTHON:-python3}
VLLM_VERSION=0.30.0
TORCH_INDEX=https://download.pytorch.org/whl/cu129
WHEEL=vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
WHEEL_URL=https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/${WHEEL//+/%2B}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
INDEX_ARGS=()
if [[ -n "${PYPI_INDEX:-}" ]]; then
  INDEX_ARGS=(--index-url "$PYPI_INDEX")
  [[ "$PYPI_INDEX" == http://* ]] && INDEX_ARGS+=(--allow-insecure-host "$(echo "$PYPI_INDEX" | cut -d/ -f3)")
fi

command -v uv >/dev/null || { echo "uv not found: pip install uv" >&2; exit 1; }
mkdir -p "$ROOT/.cache"
if [[ ! -f "$ROOT/.cache/$WHEEL" ]]; then
  echo "downloading $WHEEL"
  curl -fL --retry 5 -C - ${GITHUB_PROXY:+-x "$GITHUB_PROXY"} -o "$ROOT/.cache/$WHEEL.part" "$WHEEL_URL"
  mv "$ROOT/.cache/$WHEEL.part" "$ROOT/.cache/$WHEEL"
fi

export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-600}
uv venv -p "$PYTHON" "$VENV"
TORCH_PINS=("torch==2.13.0+cu129" "torchvision==0.28.0+cu129" "torchaudio==2.11.0+cu129")
uv pip install -p "$VENV/bin/python" --no-deps --index-url "$TORCH_INDEX" "${TORCH_PINS[@]}" "torchcodec==0.16.0+cu129"
uv pip install -p "$VENV/bin/python" "${INDEX_ARGS[@]}" "$ROOT/.cache/$WHEEL" "${TORCH_PINS[@]}" -e "$ROOT[vllm]"
# vllm pulls the CUDA 13 torchcodec from PyPI; put the cu129 build back.
uv pip install -p "$VENV/bin/python" --no-deps --reinstall --index-url "$TORCH_INDEX" "torchcodec==0.16.0+cu129"

"$VENV/bin/python" - <<'EOF'
import torch, torchcodec.decoders, vllm  # noqa: F401  (torchcodec fails to load if the CUDA build is wrong)
print(f"vllm {vllm.__version__}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
EOF
echo "installed into $VENV; run: $VENV/bin/jevlike-infer serve --model clef-flash --model-path <dir>"
