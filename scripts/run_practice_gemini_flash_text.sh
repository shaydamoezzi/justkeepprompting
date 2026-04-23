#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Using active conda environment: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
elif [ -f "${ROOT}/.venv-vlm/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv-vlm/bin/activate"
  echo "Using virtualenv: ${ROOT}/.venv-vlm"
else
  echo "No active conda env detected and .venv-vlm is missing." >&2
  exit 1
fi

API_KEY_ENV_VAR="${API_KEY_ENV_VAR:-GEMINI_API_KEY}"
if [ -z "${!API_KEY_ENV_VAR:-}" ]; then
  echo "Missing API key env var ${API_KEY_ENV_VAR}." >&2
  echo "Export it first, for example:" >&2
  echo "  export ${API_KEY_ENV_VAR}=<your-token>" >&2
  exit 1
fi

MODEL_ID="${MODEL_ID:-gemini-2.5-flash}"
QUESTION_IDS_FILE="${QUESTION_IDS_FILE:-outputs/self_hosted_smoke_gemini31_video/question_ids_single.txt}"
OUT_DIR="${OUT_DIR:-outputs/self_hosted_practice_gemini_flash_text}"
STRATEGY="${STRATEGY:-adversarial_negation}"
MAX_TURNS="${MAX_TURNS:-2}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MIN_SECONDS_BETWEEN_REQUESTS="${MIN_SECONDS_BETWEEN_REQUESTS:-3.0}"

python "${ROOT}/scripts/smoke_test_jkp_chat.py" \
  --backend gemini_native \
  --family gemini \
  --model-id "${MODEL_ID}" \
  --api-key-env-var "${API_KEY_ENV_VAR}" \
  --strategy "${STRATEGY}" \
  --max-turns "${MAX_TURNS}" \
  --temperature "${TEMPERATURE}" \
  --text-only \
  --debug-gemini-io \
  --print-raw-turns \
  --min-seconds-between-requests "${MIN_SECONDS_BETWEEN_REQUESTS}" \
  --max-api-requests 20 \
  --question-ids-file "${QUESTION_IDS_FILE}" \
  --output-dir "${OUT_DIR}"

echo
echo "Practice run complete. Inspect artifacts in ${OUT_DIR}/runs"
