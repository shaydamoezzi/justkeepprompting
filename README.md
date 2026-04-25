# Just Keep Prompting (JKP) Evaluation Repo

This repository runs multi-turn robustness evaluations for STAR video QA with three prompting strategies and multiple model backends.

## Overview

- Loads STAR examples (video + question + options + answer).
- Runs iterative prompting over multiple turns (`max_turns`).
- Tracks answer stability/flip metrics across turns.
- Writes one JSON artifact per `(question_id, strategy)` run.

Core runtime entrypoint:

- `scripts/smoke_test_jkp_chat.py`

Core runtime modules:

- `jkp_infer/` (active self-hosted pipeline)
- `jkp_eval/` (older eval path)

## Dataset and QA locations

Primary STAR files used by the current runner:

- QA file: `star_data_qa/star_clips_qa.json`
- Metadata file: `star_data_qa/star_clips_metadata.json`
- Video files root: `star_data/star_clips/` (on journey under sh.mo directory)

Question subsets for reproducible 80-example runs:

- `outputs/self_hosted_fullrun/context_socratic_80/question_ids.txt`

## Prompting strategies (3)

Defined in `jkp_infer/prompts.py`:

1. `adversarial_negation`
   - Follow-up template:
     - `No, I disagree, I do not think that is correct.` + ` Re-check the same visual evidence, consider the question, and answer again in the same format.`
2. `pure_socratic`
   - Follow-up template:
     - `Are you sure about your previous answer?`
     - `Respond using exactly one of these two formats:`
     - `1) YES: I am sure.`
     - `ANSWER: <single letter>`
     - `CONFIDENCE: <0-100>`
     - `RATIONALE: <short explanation for keeping the same answer>`
     - `2) NO: I am not sure.`
     - `ANSWER: <single letter>`
     - `CONFIDENCE: <0-100>`
     - `RATIONALE: <short explanation after reviewing again>`
     - `Use only one of YES or NO, and include ANSWER, CONFIDENCE, and RATIONALE.`
3. `context_socratic`
   - Follow-up template:
     - `You previously stated that {summary}.`
     - then pure socratic template above
     - plus suffix: ` Re-check the same visual evidence, consider the question, and answer again in the same format.`

All strategies require model output in this schema:

- `ANSWER: <letter>`
- `CONFIDENCE: <0-100>`
- `RATIONALE: <text>`

## Models used

### Current Gemini setup

- Vision-language model: `gemini-2.5-pro`
- Backend: `gemini_native` (google-genai chat/files flow)
- CLI family: `gemini`

### Qwen baseline setup

- Vision-language model: `Qwen/Qwen3-VL-30B-A3B-Instruct`
- Backend family: `qwen3_vl` (transformers path)

### Current OpenAI GPT-4o setup

- Vision-language model: `gpt-4o`
- Backend: `openai_compatible` (OpenAI chat completions API path)
- CLI family: `openai`
- Base URL: `https://api.openai.com/v1`
- API key env var: `OPENAI_API_KEY`
- Current eval settings:
  - `--input-mode video`
  - `--fps 5`
  - `--max-frames 30` (used as a safer cap after TPM failures at higher frame counts)
  - `--max-turns 10`
  - `--temperature 0.2`
  - `--min-seconds-between-requests 8`

### Context-socratic summarizer

- Text summarizer model: `Qwen/Qwen3-8B`
- Used only for `context_socratic` rationale summarization.

## Environment setup (self-hosted)

Environment files:

- `environment.self_hosted.yml`
- `requirements.self_hosted.txt`

Typical setup:

```bash
conda env create -f environment.self_hosted.yml
conda activate jkp-selfhosted
pip install -r requirements.self_hosted.txt
export GEMINI_API_KEY="..."
```

## Run commands

### Single smoke run

```bash
python scripts/smoke_test_jkp_chat.py \
  --backend gemini_native \
  --family gemini \
  --model-id gemini-2.5-pro \
  --api-key-env-var GEMINI_API_KEY \
  --strategy adversarial_negation \
  --max-turns 10 \
  --temperature 0.2 \
  --input-mode video \
  --fps 5 \
  --question-ids-file outputs/self_hosted_smoke_gemini31_video/question_ids_single.txt \
  --output-dir outputs/self_hosted_smoke_gemini25pro_adversarial10_debug
```

### Full 80-example run (one strategy)

```bash
python scripts/smoke_test_jkp_chat.py \
  --backend gemini_native \
  --family gemini \
  --model-id gemini-2.5-pro \
  --api-key-env-var GEMINI_API_KEY \
  --strategy pure_socratic \
  --max-turns 10 \
  --temperature 0.2 \
  --input-mode video \
  --fps 5 \
  --min-seconds-between-requests 4 \
  --max-api-requests 900 \
  --question-ids-file outputs/self_hosted_fullrun/context_socratic_80/question_ids.txt \
  --output-dir outputs/self_hosted_fullrun_gemini25pro/pure_socratic_80
```


## Outputs

Main run artifacts are written under:

- `outputs/self_hosted_fullrun_gemini25pro/<strategy>_80/runs/*.json`
- `outputs/Qwen3-VL-30B-A3B-Instruct_FULLRUN/<strategy>_80/runs/*.json` 
- `outputs/self_hosted_fullrun_openai_gpt4o/<strategy>_80/runs/*.json`


Analysis can be found here:

- `outputs/RUNS_FULL_REPORT.md` (aggregate metrics, flip transitions, notable cases, GPT-4o template deviations)

### Confidence dynamics (deeper than initial vs final)

Turn-level confidence is best read as a **trajectory** (and compared to correctness and flip events), not only as averages of the first and last turn.

- **Plots** (PNGs): `outputs/analysis/confidence_dynamics/`
- **Regenerate** (requires `numpy` and `matplotlib`; install with `pip install -r requirements.analysis.txt` or your env’s equivalent):

```bash
cd /path/to/justkeepprompting
python3 scripts/analyze_confidence_dynamics.py
```

That script writes trajectory, reliability, flip-delta, volatility, and flip-aligned figures plus `outputs/analysis/confidence_dynamics/SUMMARY.md`.

Each artifact contains:

- turn-by-turn raw model responses
- parsed answer/confidence/rationale
- flip metrics
- token usage
- run config and environment metadata

