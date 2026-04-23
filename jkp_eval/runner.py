from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .backends import TransformersMultimodalBackend, TransformersTextBackend, load_json
from .metrics import compute_flip_metrics
from .prompting import (
    base_system_prompt,
    build_auxiliary_summary_prompt,
    build_followup_user_text,
    build_initial_user_text,
    heuristic_summary,
    parse_answer,
)
from .star import StarExample, load_star_examples
from .video import sample_video_frames


STRATEGIES = ("adversarial_negation", "pure_socratic", "context_socratic")


@dataclass
class EvalArtifacts:
    records: list[dict[str, Any]]


def _build_initial_messages(example: StarExample, frame_paths: list[Path]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "image", "image": str(frame_path)} for frame_path in frame_paths
    ]
    content.append({"type": "text", "text": build_initial_user_text(example)})
    return [
        {"role": "system", "content": [{"type": "text", "text": base_system_prompt()}]},
        {"role": "user", "content": content},
    ]


def _make_aux_summary(
    strategy: str,
    previous_answer: str,
    auxiliary_backend: TransformersTextBackend | None,
) -> str | None:
    if strategy != "context_socratic":
        return None
    if auxiliary_backend is None:
        return heuristic_summary(previous_answer)
    prompt = build_auxiliary_summary_prompt(previous_answer)
    return auxiliary_backend.generate(prompt).strip()


def run_eval(config_path: str | Path, *, limit: int | None = None, dry_run: bool = False) -> EvalArtifacts:
    config = load_json(config_path)
    dataset_cfg = config["dataset"]
    runner_cfg = config["runner"]
    model_cfg = config["model"]
    auxiliary_cfg = config.get("auxiliary", {})

    examples = load_star_examples(
        dataset_cfg["qa_path"],
        dataset_cfg["metadata_path"],
        categories=runner_cfg.get("categories"),
        limit=limit or runner_cfg.get("limit"),
    )
    if dry_run:
        preview = []
        for example in examples[: min(len(examples), 2)]:
            preview.append(
                {
                    "question_id": example.question_id,
                    "question": example.question,
                    "choices": example.choices,
                    "gold_letter": example.answer_letter,
                    "initial_user_text": build_initial_user_text(example),
                    "strategies": {
                        name: build_followup_user_text(name, "you chose A because ...")
                        for name in STRATEGIES
                    },
                }
            )
        return EvalArtifacts(records=preview)

    backend = TransformersMultimodalBackend(model_cfg)
    auxiliary_backend = (
        TransformersTextBackend(auxiliary_cfg)
        if auxiliary_cfg.get("enabled") and auxiliary_cfg.get("kind") == "transformers_text"
        else None
    )

    frame_count = int(runner_cfg.get("frame_count", 8))
    max_turns = int(runner_cfg.get("max_turns", 10))
    output_dir = Path(runner_cfg.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for example in examples:
        frame_dir = output_dir / "frames" / example.question_id
        frame_paths = sample_video_frames(
            example.video_path,
            num_frames=frame_count,
            output_dir=frame_dir,
        )

        for strategy in STRATEGIES:
            messages = _build_initial_messages(example, frame_paths)
            turns: list[dict[str, Any]] = []
            previous_text = ""

            for turn_index in range(max_turns + 1):
                response = backend.generate(messages, frame_paths=frame_paths)
                parsed = parse_answer(response.text, len(example.choices))
                turn_record = {
                    "turn_index": turn_index,
                    "raw_text": response.text,
                    "choice_letter": parsed.choice_letter,
                    "choice_index": parsed.choice_index,
                    "confidence": parsed.confidence,
                    "rationale": parsed.rationale,
                }
                turns.append(turn_record)

                if turn_index == max_turns:
                    break

                summary = _make_aux_summary(strategy, response.text, auxiliary_backend)
                followup_text = build_followup_user_text(strategy, summary)
                messages.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": response.text}],
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": followup_text}],
                    }
                )
                previous_text = response.text

            records.append(
                {
                    "question_id": example.question_id,
                    "category": example.category,
                    "template_id": example.template_id,
                    "strategy": strategy,
                    "model_id": model_cfg["model_id"],
                    "video_path": str(example.video_path),
                    "gold_choice_index": example.answer_index,
                    "gold_choice_letter": example.answer_letter,
                    "turns": turns,
                    "metrics": compute_flip_metrics(turns, example.answer_index),
                }
            )

    results_path = output_dir / f"{Path(config_path).stem}.jsonl"
    with results_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return EvalArtifacts(records=records)

