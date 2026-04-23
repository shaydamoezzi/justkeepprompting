#!/usr/bin/env python3
"""Smoke test: Qwen3 dense LM one-sentence summary for context_socratic follow-ups."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jkp_infer.dataset import StarExample, load_star_examples
from jkp_infer.prompts import build_followup_user_prompt, summarize_previous_rationale


_SAMPLE_ANSWER = """ANSWER: C
CONFIDENCE: 78
RATIONALE: The person picks up the red object before walking through the doorway, so the interaction order supports option C over the others.
"""


def _synthetic_assistant_turn(example: StarExample) -> str:
    """Formatted model turn grounded in a real STAR row (for summarization smoke only)."""
    letter = example.answer_letter
    rationale = (
        f"From the sampled frames for this clip, the visuals best support option {letter}: "
        f"{example.question}"
    )
    return f"ANSWER: {letter}\nCONFIDENCE: 76\nRATIONALE: {rationale}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-8B",
        help="HF model id for the text summarizer (ignored with --mock).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map (e.g. "auto", "cuda:0", "cpu").',
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["bfloat16", "float16", "float32"],
        help="Torch dtype for model weights (float16 avoids CUBLAS bf16 errors on many GPUs).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Do not load a model; use a stub summarizer.",
    )
    parser.add_argument(
        "--star",
        action="store_true",
        help="Use one real STAR example (question + choices) instead of the built-in sample.",
    )
    parser.add_argument(
        "--question-id",
        default=None,
        help="When using --star, pick this question_id (default: first loaded example).",
    )
    args = parser.parse_args()

    if args.mock:

        def stub_summarizer(text: str) -> str:
            return (
                "Stub one-sentence summary: the model chose an answer based on the described "
                "visual order in the rationale."
            )

        summarizer = stub_summarizer
    else:
        if args.device_map == "cpu" and args.dtype == "bfloat16":
            print("Note: using float32 on CPU (bfloat16 CPU matmul is often unsupported).", flush=True)
            dtype = "float32"
        else:
            dtype = args.dtype
        from jkp_infer.qwen_text_summarize import QwenTextSummarizer

        summarizer = QwenTextSummarizer(
            {
                "model_id": args.model_id,
                "device_map": args.device_map,
                "dtype": dtype,
                "max_new_tokens": 96,
                "temperature": 0.0,
            }
        )

    num_choices = 6
    if args.star:
        qa_path = ROOT / "star_data" / "star_clips_qa.json"
        meta_path = ROOT / "star_data" / "star_clips_metadata.json"
        if not qa_path.is_file() or not meta_path.is_file():
            raise SystemExit(f"Missing STAR data under {ROOT / 'star_data'} (need qa + metadata JSON).")
        examples = load_star_examples(qa_path, meta_path, limit=None)
        if not examples:
            raise SystemExit("No STAR examples loaded.")
        example: StarExample | None = None
        if args.question_id:
            for ex in examples:
                if ex.question_id == args.question_id:
                    example = ex
                    break
            if example is None:
                raise SystemExit(f"No example with question_id={args.question_id!r}.")
        else:
            example = examples[0]
        previous_answer = _synthetic_assistant_turn(example)
        num_choices = len(example.choices)
        print("--- STAR example ---", flush=True)
        print(f"question_id={example.question_id} category={example.category}", flush=True)
        print(f"question={example.question!r}", flush=True)
        print(f"choices={list(example.choices)}", flush=True)
        print("--- synthetic assistant turn (RATIONALE from real question) ---", flush=True)
        print(previous_answer[:1200] + ("..." if len(previous_answer) > 1200 else ""), flush=True)
    else:
        previous_answer = _SAMPLE_ANSWER

    summary = summarize_previous_rationale(
        previous_answer,
        text_summarizer=summarizer,
        num_choices=num_choices,
    )
    followup = build_followup_user_prompt(
        "context_socratic",
        previous_answer,
        text_summarizer=summarizer,
        num_choices=num_choices,
    )

    print("--- one-sentence summary ---", flush=True)
    print(summary, flush=True)
    print("--- context_socratic follow-up user text ---", flush=True)
    print(followup[:2000] + ("..." if len(followup) > 2000 else ""), flush=True)

    assert followup.startswith("You previously stated that ")
    assert "Are you sure about your previous answer?" in followup
    assert "Use only one of YES or NO" in followup
    assert "Re-check the same visual evidence" in followup
    assert "Why is that the case" not in followup
    display = summary.strip().rstrip(".")
    assert f"You previously stated that {display}." in followup
    if args.mock:
        assert "Stub one-sentence" in summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
