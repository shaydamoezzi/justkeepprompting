from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class StarExample:
    question_id: str
    clip_path: str
    category: str
    template_id: str
    question: str
    choices: tuple[str, ...]
    answer_index: int
    video_path: Path

    @property
    def answer_letter(self) -> str:
        return chr(ord("A") + self.answer_index)


def load_star_examples(
    qa_path: str | Path,
    metadata_path: str | Path,
    *,
    categories: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[StarExample]:
    qa_path = Path(qa_path)
    metadata_path = Path(metadata_path)
    raw_examples = json.loads(qa_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    clips_root = Path(metadata["clips_root"])
    allowed_categories = {c.lower() for c in categories} if categories else None

    results: list[StarExample] = []
    for item in raw_examples:
        if allowed_categories and item["category"].lower() not in allowed_categories:
            continue
        video_path = clips_root / item["clip_path"]
        results.append(
            StarExample(
                question_id=item["question_id"],
                clip_path=item["clip_path"],
                category=item["category"],
                template_id=item["template_id"],
                question=item["question"],
                choices=tuple(item["choices"]),
                answer_index=int(item["answer_index"]),
                video_path=video_path,
            )
        )
        if limit is not None and len(results) >= limit:
            break
    return results

