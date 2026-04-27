"""
build_hf_space_data.py

One-time script that:
  1. Scans all committed run JSONs under outputs/
  2. Computes per (model × strategy) aggregate metrics
  3. Writes hf_space/data/leaderboard.json
  4. Writes hf_space/data/runs.jsonl   (compact, no raw video paths)

Run from the repo root:
    python3 scripts/build_hf_space_data.py
"""

from __future__ import annotations

import json
import re
import glob
import os
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUT_DIR = REPO_ROOT / "hf_space" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── model display labels ────────────────────────────────────────────────────
MODEL_LABELS = {
    "Qwen/Qwen3-VL-30B-A3B-Instruct": "Qwen3-VL-30B",
    "gpt-4o": "GPT-4o",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "OpenGVLab/InternVL3_5-30B-A3B-HF": "InternVL3.5-30B",
}

# ── helpers ─────────────────────────────────────────────────────────────────

_Q_PATTERN = re.compile(
    r"Question:\s*(.+?)\nOptions:\n(.*?)$",
    re.DOTALL,
)
_OPT_PATTERN = re.compile(r"^[A-D]\.\s*(.+)$", re.MULTILINE)


def parse_question_options(user_text: str) -> tuple[str, list[str]]:
    m = _Q_PATTERN.search(user_text)
    if not m:
        return "", []
    question = m.group(1).strip()
    opts_raw = m.group(2).strip()
    options = _OPT_PATTERN.findall(opts_raw)
    return question, options


def extract_conversation(chat_messages: list[dict]) -> list[dict]:
    """Return [{role, content}] with video/image parts stripped out."""
    conv = []
    for msg in chat_messages:
        if msg["role"] == "system":
            continue
        content = msg["content"]
        if isinstance(content, list):
            texts = [p["text"] for p in content if p.get("type") == "text"]
            text = "\n".join(texts).strip()
        else:
            text = str(content).strip()
        if text:
            conv.append({"role": msg["role"], "content": text})
    return conv


def compact_run(path: Path) -> dict | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}", file=sys.stderr)
        return None

    cfg = d.get("run_metadata", {}).get("run_config", {}).get("cli_args", {})
    ds = d.get("run_metadata", {}).get("dataset", {})
    metrics = d.get("metrics", {})

    model_id: str = cfg.get("model_id") or cfg.get("family") or "unknown"
    # normalise gemini model id
    if model_id in ("gemini", "gemini_native"):
        model_id = "gemini-2.5-pro"

    strategy: str = d.get("strategy") or cfg.get("strategy") or "unknown"
    question_id: str = d.get("question_id", path.stem)

    # parse question + options from first user message
    question, options = "", []
    for msg in d.get("chat_messages", []):
        if msg["role"] == "user":
            txt = ""
            if isinstance(msg["content"], list):
                texts = [p["text"] for p in msg["content"] if p.get("type") == "text"]
                txt = " ".join(texts)
            else:
                txt = str(msg["content"])
            question, options = parse_question_options(txt)
            if question:
                break

    turns = [
        {
            "turn_index": t.get("turn_index"),
            "choice_letter": t.get("choice_letter"),
            "confidence": t.get("confidence"),
            "sure_status": t.get("sure_status"),
            "rationale": t.get("rationale", ""),
            "changed_from_previous": t.get("changed_from_previous", False),
            "parse_success": t.get("parse_success", True),
        }
        for t in d.get("turns", [])
    ]

    conv = extract_conversation(d.get("chat_messages", []))

    number_of_flips = metrics.get("number_of_flips", 0)
    # bool: did any flip occur?
    had_flip = number_of_flips > 0

    return {
        "run_id": f"{MODEL_LABELS.get(model_id, model_id)}__{strategy}__{question_id}",
        "model_id": model_id,
        "model_label": MODEL_LABELS.get(model_id, model_id),
        "strategy": strategy,
        "question_id": question_id,
        "category": ds.get("category") or question_id.split("_")[0],
        "template_id": ds.get("template_id") or question_id.split("_")[1] if "_" in question_id else "",
        "answer_letter": ds.get("answer_letter", ""),
        "answer_index": ds.get("answer_index"),
        "question": question,
        "options": options,
        "initial_correct": metrics.get("initial_correct"),
        "final_correct": metrics.get("final_correct"),
        "number_of_flips": number_of_flips,
        "had_flip": had_flip,
        "turn_of_first_change": metrics.get("turn_of_first_change"),
        "turns": turns,
        "conversation": conv,
    }


# ── discover run files ───────────────────────────────────────────────────────

def discover_runs() -> list[Path]:
    """Return all run JSON files from outputs/."""
    paths: list[Path] = []

    # GPT-4o and Gemini: outputs/<MODEL>_FULLRUN/<strategy>_80/runs/*.json
    for p in OUTPUTS_DIR.glob("*_FULLRUN/*/runs/*.json"):
        paths.append(p)

    # Qwen adversarial: has a runs/ subdir
    for p in OUTPUTS_DIR.glob("Qwen3-VL-30B-A3B-Instruct_FULLRUN/adversarial_negation_80/runs/*.json"):
        if p not in paths:
            paths.append(p)

    # Qwen context + pure: JSON files directly in the strategy dir (no runs/ subdir)
    for strategy_dir in [
        "Qwen3-VL-30B-A3B-Instruct_FULLRUN/context_socratic_80",
        "Qwen3-VL-30B-A3B-Instruct_FULLRUN/pure_socratic_80",
    ]:
        for p in (OUTPUTS_DIR / strategy_dir).glob("*.json"):
            paths.append(p)

    # InternVL self_hosted runs
    for p in OUTPUTS_DIR.glob("self_hosted_fullrun/*/runs/*.json"):
        paths.append(p)

    return sorted(set(paths))


# ── aggregate leaderboard ───────────────────────────────────────────────────

def build_leaderboard(runs: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {
        "total": 0,
        "initial_correct": 0,
        "final_correct": 0,
        "flipped": 0,
        "total_flips": 0,
        "conf_deltas": [],
    })

    for r in runs:
        key = (r["model_label"], r["strategy"])
        g = groups[key]
        g["total"] += 1
        if r["initial_correct"]:
            g["initial_correct"] += 1
        if r["final_correct"]:
            g["final_correct"] += 1
        if r["had_flip"]:
            g["flipped"] += 1
        g["total_flips"] += r["number_of_flips"]

        # confidence delta: final conf - initial conf (if both present)
        turns = r["turns"]
        if len(turns) >= 2:
            c0 = turns[0].get("confidence")
            cf = turns[-1].get("confidence")
            if c0 is not None and cf is not None:
                g["conf_deltas"].append(cf - c0)

    rows = []
    for (model_label, strategy), g in groups.items():
        n = g["total"]
        if n == 0:
            continue
        initial_acc = 100 * g["initial_correct"] / n
        final_acc = 100 * g["final_correct"] / n
        flip_rate = 100 * g["flipped"] / n
        avg_flips = g["total_flips"] / n
        conf_delta = (
            round(sum(g["conf_deltas"]) / len(g["conf_deltas"]), 2)
            if g["conf_deltas"]
            else None
        )
        gtt_score = round(final_acc * (1 - flip_rate / 100), 2)

        rows.append({
            "model_label": model_label,
            "strategy": strategy,
            "n_runs": n,
            "initial_acc": round(initial_acc, 1),
            "final_acc": round(final_acc, 1),
            "flip_rate": round(flip_rate, 1),
            "avg_flips": round(avg_flips, 2),
            "conf_delta": conf_delta,
            "gtt_score": gtt_score,
        })

    rows.sort(key=lambda r: r["gtt_score"], reverse=True)
    return rows


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Discovering run JSON files…")
    paths = discover_runs()
    print(f"  Found {len(paths)} files")

    print("Parsing runs…")
    runs: list[dict] = []
    for p in paths:
        r = compact_run(p)
        if r:
            runs.append(r)
    print(f"  Parsed {len(runs)} valid runs")

    # ── write runs.jsonl ────────────────────────────────────────────────────
    runs_path = OUT_DIR / "runs.jsonl"
    with open(runs_path, "w") as f:
        for r in runs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {runs_path}")

    # ── write leaderboard.json ──────────────────────────────────────────────
    lb = build_leaderboard(runs)
    lb_path = OUT_DIR / "leaderboard.json"
    with open(lb_path, "w") as f:
        json.dump(lb, f, indent=2, ensure_ascii=False)
    print(f"Wrote {lb_path} ({len(lb)} rows)")


if __name__ == "__main__":
    main()
