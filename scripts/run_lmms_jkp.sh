#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LMMS_DIR="${ROOT}/vendor/lmms-eval"

if [ ! -d "${LMMS_DIR}" ]; then
  echo "Missing ${LMMS_DIR}. Clone lmms-eval first." >&2
  exit 1
fi

if [ ! -d "${ROOT}/.venv-vlm" ]; then
  echo "Missing ${ROOT}/.venv-vlm. Run scripts/setup_vlm_env.sh first." >&2
  exit 1
fi

source "${ROOT}/.venv-vlm/bin/activate"

MODEL_FAMILY="${1:-qwen3_vl}"
TASK_NAME="${2:-jkp_star_negation}"
MODEL_ARGS="${3:-pretrained=Qwen/Qwen3-VL-4B-Instruct,device_map=auto,torch_dtype=float16,max_num_frames=8,fps=1,attn_implementation=sdpa,system_prompt=You are a strict video QA evaluator. Always follow the requested output format.}"

cd "${LMMS_DIR}"
PYTHONPATH="${LMMS_DIR}:${PYTHONPATH:-}" python -m lmms_eval \
  --model "${MODEL_FAMILY}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASK_NAME}" \
  --batch_size 1 \
  --log_samples \
  --agentic_trace_mode full \
  --output_path "${ROOT}/outputs/lmms_eval" \
  --verbosity INFO \
  --gen_kwargs "temperature=0.2,max_new_tokens=256"
