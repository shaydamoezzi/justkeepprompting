from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .dataset import StarExample
from .prompts import build_followup_user_prompt, build_initial_user_prompt, build_system_prompt


Message = dict[str, object]


@dataclass
class ConversationState:
    example: StarExample
    strategy: str
    frame_paths: list[Path]
    messages: list[Message]
    text_summarizer: Callable[[str], str] | None = None


def build_initial_conversation(
    example: StarExample,
    strategy: str,
    frame_paths: list[Path],
    *,
    video_path: Path | None = None,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    text_summarizer: Callable[[str], str] | None = None,
    include_visual: bool = True,
) -> ConversationState:
    user_content: list[dict[str, object]] = []
    if include_visual and video_path is not None:
        video_block: dict[str, object] = {"type": "video", "video": str(video_path)}
        if video_fps is not None:
            video_block["fps"] = video_fps
        if video_max_frames is not None:
            video_block["max_frames"] = video_max_frames
        user_content.append(video_block)
    elif include_visual:
        user_content.extend({"type": "image", "image": str(frame_path)} for frame_path in frame_paths)
    user_content.append({"type": "text", "text": build_initial_user_prompt(example)})
    messages: list[Message] = [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt()}]},
        {"role": "user", "content": user_content},
    ]
    return ConversationState(
        example=example,
        strategy=strategy,
        frame_paths=frame_paths,
        messages=messages,
        text_summarizer=text_summarizer,
    )


def append_assistant_message(state: ConversationState, response_text: str) -> None:
    state.messages.append(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}],
        }
    )


def append_followup_user_message(state: ConversationState, previous_answer: str) -> None:
    state.messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": build_followup_user_prompt(
                        state.strategy,
                        previous_answer,
                        text_summarizer=state.text_summarizer,
                        num_choices=len(state.example.choices),
                    ),
                }
            ],
        }
    )

