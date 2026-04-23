from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import time
from collections.abc import Callable
from typing import Any

from .backends import ChatBackend
from .conversation import (
    append_assistant_message,
    append_followup_user_message,
    build_initial_conversation,
)
from .dataset import StarExample
from .metrics import compute_flip_metrics
from .prompts import parse_answer
from .video import probe_video_metadata, sample_video_frames


@dataclass
class ExampleRun:
    question_id: str
    strategy: str
    video_path: str
    frame_paths: list[str]
    turns: list[dict[str, Any]]
    metrics: dict[str, Any]
    token_usage: dict[str, int | None]
    run_metadata: dict[str, Any]
    chat_messages: list[dict[str, Any]]


def run_example_chat(
    *,
    example: StarExample,
    strategy: str,
    backend: ChatBackend,
    max_turns: int,
    frame_count: int | None,
    frame_fps: float | None = None,
    frame_max: int | None = None,
    input_mode: str = "frames",
    run_config: dict[str, Any] | None = None,
    output_dir: str | Path,
    text_summarizer: Callable[[str], str] | None = None,
    include_visual: bool = True,
) -> ExampleRun:
    run_start = datetime.now(timezone.utc)
    run_start_perf = time.perf_counter()
    output_dir = Path(output_dir)
    video_metadata = probe_video_metadata(example.video_path)
    prompt_hash = hashlib.sha256(example.question.encode("utf-8")).hexdigest()
    options_hash = hashlib.sha256("\n".join(example.choices).encode("utf-8")).hexdigest()
    video_sampling = {
        "input_mode": input_mode,
        "include_visual": include_visual,
        "requested_fps": frame_fps,
        "requested_max_frames": frame_max,
        "requested_num_frames": frame_count,
        "effective_frame_count": None,
        "decoder_backend": "qwen_vl_utils/decord_or_auto" if input_mode == "video" else "opencv",
    }
    if input_mode == "video" and include_visual:
        frame_paths: list[Path] = []
        state = build_initial_conversation(
            example,
            strategy=strategy,
            frame_paths=frame_paths,
            video_path=Path(example.video_path),
            video_fps=frame_fps,
            video_max_frames=frame_max,
            text_summarizer=text_summarizer,
            include_visual=include_visual,
        )
    elif include_visual:
        frame_dir = output_dir / "frames" / example.question_id / strategy
        frame_paths = sample_video_frames(
            example.video_path,
            num_frames=frame_count,
            fps=frame_fps,
            max_frames=frame_max,
            output_dir=frame_dir,
        )
        state = build_initial_conversation(
            example,
            strategy=strategy,
            frame_paths=frame_paths,
            text_summarizer=text_summarizer,
            include_visual=include_visual,
        )
    else:
        frame_paths = []
        state = build_initial_conversation(
            example,
            strategy=strategy,
            frame_paths=frame_paths,
            text_summarizer=text_summarizer,
            include_visual=include_visual,
        )
    video_sampling["effective_frame_count"] = (
        len(frame_paths) if (input_mode == "frames" and include_visual) else frame_max
    )
    turns: list[dict[str, Any]] = []
    prompt_tokens_sum = 0
    completion_tokens_sum = 0
    total_tokens_sum = 0
    previous_choice_letter: str | None = None

    for turn_index in range(max_turns + 1):
        turn_start_perf = time.perf_counter()
        response = backend.generate(state, turn_index)
        generate_ms = int((time.perf_counter() - turn_start_perf) * 1000)
        parse_start_perf = time.perf_counter()
        parsed = parse_answer(response.text, len(example.choices))
        parse_ms = int((time.perf_counter() - parse_start_perf) * 1000)
        turn_token_usage = response.token_usage
        if turn_token_usage:
            prompt_tokens_sum += int(turn_token_usage.get("prompt_tokens", 0))
            completion_tokens_sum += int(turn_token_usage.get("completion_tokens", 0))
            total_tokens_sum += int(turn_token_usage.get("total_tokens", 0))
        changed_from_previous = previous_choice_letter is not None and parsed.choice_letter != previous_choice_letter
        turns.append(
            {
                "turn_index": turn_index,
                "raw_text": response.text,
                "choice_letter": parsed.choice_letter,
                "choice_index": parsed.choice_index,
                "confidence": parsed.confidence,
                "sure_status": parsed.sure_status,
                "rationale": parsed.rationale,
                "parse_success": parsed.choice_letter is not None,
                "changed_from_previous": changed_from_previous,
                "timing_ms": {
                    "generate_ms": generate_ms,
                    "parse_ms": parse_ms,
                    "turn_total_ms": int((time.perf_counter() - turn_start_perf) * 1000),
                },
                "token_usage": turn_token_usage,
                "cumulative_token_usage": {
                    "prompt_tokens": prompt_tokens_sum if prompt_tokens_sum > 0 else None,
                    "completion_tokens": completion_tokens_sum if completion_tokens_sum > 0 else None,
                    "total_tokens": total_tokens_sum if total_tokens_sum > 0 else None,
                },
            }
        )
        previous_choice_letter = parsed.choice_letter
        append_assistant_message(state, response.text)
        if turn_index < max_turns:
            append_followup_user_message(state, response.text)

    metrics = compute_flip_metrics(turns, example.answer_index)
    run_end = datetime.now(timezone.utc)
    run_wall_ms = int((time.perf_counter() - run_start_perf) * 1000)
    return ExampleRun(
        question_id=example.question_id,
        strategy=strategy,
        video_path=str(example.video_path),
        frame_paths=[str(path) for path in frame_paths],
        turns=turns,
        metrics=metrics,
        token_usage={
            "prompt_tokens": prompt_tokens_sum if prompt_tokens_sum > 0 else None,
            "completion_tokens": completion_tokens_sum if completion_tokens_sum > 0 else None,
            "total_tokens": total_tokens_sum if total_tokens_sum > 0 else None,
        },
        run_metadata={
            "timestamps": {
                "started_at_utc": run_start.isoformat(),
                "ended_at_utc": run_end.isoformat(),
                "run_wall_ms": run_wall_ms,
            },
            "dataset": {
                "question_id": example.question_id,
                "category": example.category,
                "template_id": example.template_id,
                "question_hash_sha256": prompt_hash,
                "options_hash_sha256": options_hash,
                "answer_index": example.answer_index,
                "answer_letter": example.answer_letter,
            },
            "video": video_metadata,
            "video_sampling": video_sampling,
            "environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "run_config": run_config or {},
        },
        chat_messages=state.messages,
    )


def write_run_artifact(run: ExampleRun, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(run)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path

