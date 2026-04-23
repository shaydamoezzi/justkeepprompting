# Gemini Self-Hosted Eval Quickstart

This guide explains how to set up a fresh environment and launch the Gemini evaluation runs used in this repository.

## 1) Create and activate the conda environment

From the repo root:

```bash
conda env create -f environment.self_hosted.yml
conda activate jkp-selfhosted
```

If the env already exists:

```bash
conda activate jkp-selfhosted
```

## 2) Install Python dependencies

```bash
pip install -r requirements.self_hosted.txt
```

## 3) Set your Gemini API key

```bash
export GEMINI_API_KEY="PASTE_YOUR_KEY_HERE"
```

To make this persistent, add it to your shell profile (for example `~/.bashrc`).

## 4) Run a single-example smoke test (recommended)

Use this first to verify model I/O formatting and turn-by-turn behavior:

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
  --min-seconds-between-requests 8 \
  --max-api-requests 40 \
  --question-ids-file outputs/self_hosted_smoke_gemini31_video/question_ids_single.txt \
  --output-dir outputs/self_hosted_smoke_gemini25pro_adversarial10_debug \
  --debug-gemini-io \
  --print-raw-turns
```

Smoke-test artifact path:

`outputs/self_hosted_smoke_gemini25pro_adversarial10_debug/runs/Feasibility_T2_1027_adversarial_negation.json`

## 5) Launch the full adversarial set (80 examples)

This runs `adversarial_negation` over the full question ID list:

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
  --min-seconds-between-requests 8 \
  --max-api-requests 900 \
  --question-ids-file outputs/self_hosted_fullrun/context_socratic_80/question_ids.txt \
  --output-dir outputs/self_hosted_fullrun_gemini25pro/adversarial_negation_80
```

## Notes

- `--family gemini` is the current name; `gemini_openai` remains as a backward-compatible alias.
- Video mode uses uploaded file references and follow-up chat turns, not repeated full multimodal replay per turn.
- `--temperature 0.2` matches prior Qwen run settings for consistency.
- `--min-seconds-between-requests` helps avoid burst-related limits.
