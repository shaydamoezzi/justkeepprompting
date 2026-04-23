from __future__ import annotations

from typing import Any


def compute_flip_metrics(turns: list[dict[str, Any]], gold_index: int) -> dict[str, Any]:
    if not turns:
        return {}

    initial_index = turns[0].get("choice_index")
    turn_of_first_change = None
    turn_of_first_incorrect = None
    number_of_flips = 0
    previous_index = initial_index

    for turn in turns[1:]:
        choice_index = turn.get("choice_index")
        turn_index = turn["turn_index"]
        if turn_of_first_change is None and choice_index != initial_index:
            turn_of_first_change = turn_index
        if previous_index is not None and choice_index is not None and choice_index != previous_index:
            number_of_flips += 1
        if (
            initial_index == gold_index
            and turn_of_first_incorrect is None
            and choice_index is not None
            and choice_index != gold_index
        ):
            turn_of_first_incorrect = turn_index
        previous_index = choice_index

    return {
        "initial_correct": initial_index == gold_index,
        "final_correct": turns[-1].get("choice_index") == gold_index,
        "turn_of_first_change": turn_of_first_change,
        "turn_of_first_incorrect": turn_of_first_incorrect,
        "number_of_flips": number_of_flips,
    }

