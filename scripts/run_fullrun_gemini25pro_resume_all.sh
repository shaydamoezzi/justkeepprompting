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
  echo "Activate your conda env first (example: conda activate jkp-selfhosted)." >&2
  exit 1
fi

QUESTION_IDS_FILE="${QUESTION_IDS_FILE:-outputs/self_hosted_fullrun/context_socratic_80/question_ids.txt}"
OUT_ROOT="${OUT_ROOT:-outputs/self_hosted_fullrun_gemini25pro}"
MODEL_ID="${MODEL_ID:-gemini-2.5-pro}"
API_KEY_ENV_VAR="${API_KEY_ENV_VAR:-GEMINI_API_KEY}"
MAX_API_REQUESTS_PER_STRATEGY="${MAX_API_REQUESTS_PER_STRATEGY:-900}"
MIN_SECONDS_BETWEEN_REQUESTS="${MIN_SECONDS_BETWEEN_REQUESTS:-4.0}"
MAX_TURNS="${MAX_TURNS:-10}"
FPS="${FPS:-5}"
MAX_FRAMES="${MAX_FRAMES:-}"
TEMPERATURE="${TEMPERATURE:-0.2}"

if [ -z "${!API_KEY_ENV_VAR:-}" ]; then
  echo "Missing API key env var ${API_KEY_ENV_VAR}." >&2
  echo "Export it first, for example:" >&2
  echo "  export ${API_KEY_ENV_VAR}=<your-token>" >&2
  exit 1
fi

mkdir -p "${ROOT}/${OUT_ROOT}"

for strategy in adversarial_negation pure_socratic context_socratic; do
  output_dir="${OUT_ROOT}/${strategy}_80"
  runs_dir="${output_dir}/runs"
  remaining_ids_file="${output_dir}/question_ids_remaining.txt"
  mkdir -p "${ROOT}/${runs_dir}"

  python - <<'PY' "$ROOT" "$QUESTION_IDS_FILE" "$runs_dir" "$strategy" "$remaining_ids_file"
import sys
from pathlib import Path

root = Path(sys.argv[1])
master_ids_path = root / sys.argv[2]
runs_dir = root / sys.argv[3]
strategy = sys.argv[4]
remaining_path = root / sys.argv[5]

wanted = [
    line.strip()
    for line in master_ids_path.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.startswith("#")
]

done = set()
if runs_dir.exists():
    suffix = f"_{strategy}.json"
    for artifact in runs_dir.glob(f"*{suffix}"):
        stem = artifact.name[: -len(suffix)]
        done.add(stem)

remaining = [qid for qid in wanted if qid not in done]
remaining_path.parent.mkdir(parents=True, exist_ok=True)
remaining_path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
print(f"{strategy}: done={len(done)} remaining={len(remaining)} total={len(wanted)}")
PY

  if [ ! -s "${ROOT}/${remaining_ids_file}" ]; then
    echo "Skipping ${strategy}; all question IDs already completed."
    continue
  fi

  echo "Running strategy=${strategy} with remaining IDs from ${remaining_ids_file}"

  cmd=(
    python "${ROOT}/scripts/smoke_test_jkp_chat.py"
    --backend gemini_native
    --family gemini
    --model-id "${MODEL_ID}"
    --api-key-env-var "${API_KEY_ENV_VAR}"
    --min-seconds-between-requests "${MIN_SECONDS_BETWEEN_REQUESTS}"
    --max-api-requests "${MAX_API_REQUESTS_PER_STRATEGY}"
    --strategy "${strategy}"
    --max-turns "${MAX_TURNS}"
    --temperature "${TEMPERATURE}"
    --input-mode video
    --fps "${FPS}"
    --question-ids-file "${remaining_ids_file}"
    --output-dir "${output_dir}"
  )

  if [ -n "${MAX_FRAMES}" ]; then
    cmd+=(--max-frames "${MAX_FRAMES}")
  fi

  "${cmd[@]}"
done

echo
echo "All requested strategies complete under ${OUT_ROOT}/"
