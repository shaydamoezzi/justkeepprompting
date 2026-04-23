#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Environment bootstrap:
# - If already running inside conda (e.g., jkp-selfhosted), keep current env.
# - Else, activate .venv-vlm if available.
if [ -n "${CONDA_PREFIX:-}" ]; then
  echo "Using active conda environment: ${CONDA_DEFAULT_ENV:-unknown} (${CONDA_PREFIX})"
elif [ -f "${ROOT}/.venv-vlm/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${ROOT}/.venv-vlm/bin/activate"
  echo "Using virtualenv: ${ROOT}/.venv-vlm"
else
  echo "No active conda env detected and .venv-vlm is missing." >&2
  echo "Activate your conda env first (example: conda activate jkp-selfhosted)," >&2
  echo "or create the venv with scripts/setup_vlm_env.sh." >&2
  exit 1
fi

QUESTION_IDS_FILE="${QUESTION_IDS_FILE:-outputs/self_hosted_fullrun/context_socratic_80/question_ids.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/self_hosted_fullrun_gemini31}"
MODEL_ID="${MODEL_ID:-gemini-3.1-pro-preview}"
API_BASE_URL="${API_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
API_KEY_ENV_VAR="${API_KEY_ENV_VAR:-GEMINI_API_KEY}"
MAX_API_REQUESTS_PER_STRATEGY="${MAX_API_REQUESTS_PER_STRATEGY:-900}"
MIN_SECONDS_BETWEEN_REQUESTS="${MIN_SECONDS_BETWEEN_REQUESTS:-4.0}"
MAX_TURNS="${MAX_TURNS:-10}"
FPS="${FPS:-5}"
# Leave empty to sample uniformly at FPS across full video duration.
MAX_FRAMES="${MAX_FRAMES:-}"
TEMPERATURE="${TEMPERATURE:-1.0}"

if [ -z "${!API_KEY_ENV_VAR:-}" ]; then
  echo "Missing API key env var ${API_KEY_ENV_VAR}." >&2
  echo "Export it first, for example:" >&2
  echo "  export ${API_KEY_ENV_VAR}=<your-token>" >&2
  exit 1
fi

mkdir -p "${ROOT}/${OUT_ROOT}"

for strategy in adversarial_negation pure_socratic context_socratic; do
  output_dir="${OUT_ROOT}/${strategy}_80"
  mkdir -p "${ROOT}/${output_dir}"
  echo "Running strategy=${strategy} output_dir=${output_dir}"

  cmd=(
    python "${ROOT}/scripts/smoke_test_jkp_chat.py"
    --backend gemini_native
    --family gemini
    --model-id "${MODEL_ID}"
    --api-base-url "${API_BASE_URL}"
    --api-key-env-var "${API_KEY_ENV_VAR}"
    --min-seconds-between-requests "${MIN_SECONDS_BETWEEN_REQUESTS}"
    --max-api-requests "${MAX_API_REQUESTS_PER_STRATEGY}"
    --strategy "${strategy}"
    --max-turns "${MAX_TURNS}"
    --temperature "${TEMPERATURE}"
    --input-mode video
    --fps "${FPS}"
    --question-ids-file "${QUESTION_IDS_FILE}"
    --output-dir "${output_dir}"
  )

  if [ -n "${MAX_FRAMES}" ]; then
    cmd+=(--max-frames "${MAX_FRAMES}")
  fi

  "${cmd[@]}"
done

echo
echo "Done. Results written under ${OUT_ROOT}/"
