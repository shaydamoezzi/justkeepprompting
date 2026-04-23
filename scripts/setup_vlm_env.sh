#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT}/.venv-vlm"
PYTHON_BIN="${PYTHON_BIN:-}"

python_supports_sqlite() {
  local candidate="$1"
  "${candidate}" - <<'PY' >/dev/null 2>&1
import sqlite3
PY
}

python_supports_transformers() {
  local candidate="$1"
  "${candidate}" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3.12 python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_supports_transformers "$(command -v "${candidate}")"; then
      PYTHON_BIN="$(command -v "${candidate}")"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ]; then
  echo "Could not find a Python interpreter new enough for the transformers build." >&2
  exit 1
fi

rm -rf "${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if ! python_supports_sqlite "${PYTHON_BIN}"; then
  pip install pysqlite3-binary
  cat > "${VENV_DIR}/lib/$(python - <<'PY'
import sys
print(f'python{sys.version_info.major}.{sys.version_info.minor}')
PY
)/site-packages/sitecustomize.py" <<'PY'
import sys

try:
    import pysqlite3
except Exception:
    pass
else:
    sys.modules.setdefault("sqlite3", pysqlite3)
    try:
        import pysqlite3._sqlite3 as _sqlite3
    except Exception:
        pass
    else:
        sys.modules.setdefault("_sqlite3", _sqlite3)
PY
fi

# Use a Torch wheel set that still includes Volta (sm70) support for the Tesla V100s
# on this machine. The 2.11.0+cu128 wheels drop the kernels we need.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Qwen3-VL requires a bleeding-edge transformers build; InternVL3.5 requires >= 4.52.1.
pip install git+https://github.com/huggingface/transformers
pip install accelerate bitsandbytes sentencepiece safetensors huggingface_hub openai
pip install qwen-vl-utils==0.0.14 opencv-python-headless

if [ -d "${ROOT}/vendor/lmms-eval" ]; then
  pip install -e "${ROOT}/vendor/lmms-eval"
fi

echo
echo "Environment ready at ${VENV_DIR}"
echo "Python used: ${PYTHON_BIN}"
echo "Activate it with: source ${VENV_DIR}/bin/activate"
