#!/usr/bin/env bash
# Run self-hosted InternVL (transformers) on all 80 benchmark questions, for each
# of the three JKP strategies, with four parallel workers (one physical GPU each).
# Uses --input-mode video so frames are decoded in memory for inference; no
# per-run JPEGs under output_dir/frames/. See jkp_infer/pipeline.py.
#
# Usage (from repo root):
#   bash scripts/run_self_hosted_full_4gpu.sh
#
# Optional env:
#   JKP_PYTHON  Python binary (default: jkp-selfhosted miniforge env)
#   OUT_DIR     Output base (default: outputs/self_hosted_InternVL30B_80q_video)
#   MAX_TURNS   (default: 1)
#   MAX_FRAMES  Cap for video sampling (default: 30)
#   FPS         Sampling rate for in-memory video decode (default: 1.0)
#   SLICES      Number of question-ID shards = parallel jobs (default: 4 = one job per GPU)
#   GPUS        Comma list of host GPU index used as CUDA_VISIBLE_DEVICES for worker i (default: 0,1,2,3)
#   STRATEGIES  Space-separated (default: adversarial_negation pure_socratic context_socratic)
#   EXTRA       Extra args passed to every smoke test (e.g. '--max-memory-per-gpu-gib 40')
#   CONTEXT_EXTRA  Only for context_socratic (default: none). Set to
#                  '--skip-text-summarizer' if a single-GPU job OOMs with Qwen3 text.
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

JKP_PYTHON="${JKP_PYTHON:-/home/sh.mo/.miniforge3/envs/jkp-selfhosted/bin/python}"
OUT_DIR="${OUT_DIR:-outputs/self_hosted_InternVL30B_80q_video}"
MAX_TURNS="${MAX_TURNS:-1}"
MAX_FRAMES="${MAX_FRAMES:-30}"
FPS="${FPS:-1.0}"
SLICES="${SLICES:-4}"
GPUS="${GPUS:-0,1,2,3}"
STRATEGIES="${STRATEGIES:-adversarial_negation pure_socratic context_socratic}"
ID_FILE="${ID_FILE:-$ROOT/scripts/self_hosted_80q_question_ids.txt}"
MODEL_PRESET="${MODEL_PRESET:-internvl3_5_30b_a3b}"

if [[ ! -f "$ID_FILE" ]]; then
  echo "Question id file not found: $ID_FILE" >&2
  exit 1
fi
if [[ ! -x "$JKP_PYTHON" && ! -f "$JKP_PYTHON" ]]; then
  echo "Set JKP_PYTHON to your python; not found: $JKP_PYTHON" >&2
  exit 1
fi

mapfile -t GPU_ARR < <(echo "$GPUS" | tr ',' '\n' | sed '/^$/d')
if [[ "${#GPU_ARR[@]}" -ne "$SLICES" ]]; then
  echo "GPUS must list exactly SLICES ($SLICES) device ids; got ${#GPU_ARR[@]}" >&2
  exit 1
fi

SHARD_DIR="${OUT_DIR}/_shards_$$"
mkdir -p "$OUT_DIR" "$SHARD_DIR"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

n_ids="$(awk '!/^#/ && NF {print}' "$ID_FILE" | wc -l)"
n_ids="${n_ids// /}"
lines_per_shard=$(( (n_ids + SLICES - 1) / SLICES ))
# Strip comments/blank lines, split into N roughly equal line counts.
awk '!/^#/ && NF {print}' "$ID_FILE" | split -l "$lines_per_shard" - "$SHARD_DIR/shard_"

shards=("$SHARD_DIR"/shard_*)
if [[ ${#shards[@]} -ne $SLICES ]]; then
  echo "Expected $SLICES shard files, got ${#shards[@]} (check $n_ids ids vs SLICES=$SLICES)" >&2
  exit 1
fi

for strategy in $STRATEGIES; do
  echo "=== strategy=$strategy ==="
  pids=()
  for i in $(seq 0 $((SLICES - 1))); do
    shard="${shards[$i]}"
    gpu="${GPU_ARR[$i]}"
    log="${LOG_DIR}/${strategy}_gpu${gpu}.log"
    ctx_flag=()
    if [[ "$strategy" == "context_socratic" && -n "${CONTEXT_EXTRA:-}" ]]; then
      read -r -a ctx_flag <<< "$CONTEXT_EXTRA"
    fi
    extra_a=()
    if [[ -n "${EXTRA:-}" ]]; then
      read -r -a extra_a <<< "$EXTRA"
    fi
    (
      "$JKP_PYTHON" -u scripts/smoke_test_jkp_chat.py \
        --backend transformers \
        --model-preset "$MODEL_PRESET" \
        --cuda-visible-devices "$gpu" \
        --skip-v100-validation \
        --input-mode video \
        --fps "$FPS" \
        --max-frames "$MAX_FRAMES" \
        --max-turns "$MAX_TURNS" \
        --strategy "$strategy" \
        --question-ids-file "$shard" \
        --output-dir "$OUT_DIR" \
        "${ctx_flag[@]}" \
        "${extra_a[@]}" \
    ) >"$log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || { echo "Worker pid=$pid failed; see logs in $LOG_DIR" >&2; exit 1; }
  done
  echo "=== strategy $strategy done ==="
done

rm -rf "$SHARD_DIR"
echo "All runs finished. Artifacts: $OUT_DIR/runs/"
