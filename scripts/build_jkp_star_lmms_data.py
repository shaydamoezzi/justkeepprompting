#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jkp_eval.star import load_star_examples


STRATEGIES = (
    "adversarial_negation",
    "pure_socratic",
    "context_socratic",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local JSONL files for the lmms-eval STAR task.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=10)
    args = parser.parse_args()

    examples = load_star_examples(
        ROOT / "star_data/star_clips_qa.json",
        ROOT / "star_data/star_clips_metadata.json",
        limit=args.limit,
    )
    output_dir = ROOT / "vendor/lmms-eval/lmms_eval/tasks/jkp_star/data"
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_to_filename = {
        "adversarial_negation": "jkp_star_negation.jsonl",
        "pure_socratic": "jkp_star_pure_socratic.jsonl",
        "context_socratic": "jkp_star_context_socratic.jsonl",
    }

    for strategy in STRATEGIES:
        out_path = output_dir / strategy_to_filename[strategy]
        with out_path.open("w") as handle:
            for example in examples:
                record = {
                    "question_id": example.question_id,
                    "video_path": str(example.video_path),
                    "category": example.category,
                    "template_id": example.template_id,
                    "question": example.question,
                    "choices": list(example.choices),
                    "answer_index": example.answer_index,
                    "answer_letter": example.answer_letter,
                    "strategy": strategy,
                    "max_turns": args.max_turns,
                }
                handle.write(json.dumps(record) + "\n")
        print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

