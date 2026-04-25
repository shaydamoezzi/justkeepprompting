#!/usr/bin/env python3
"""
Rich turn-level confidence dynamics from JKP run JSONs: trajectories, calibration,
flip-adjacent deltas, within-run volatility. Writes PNG plots + SUMMARY.md.

Usage:
  python scripts/analyze_confidence_dynamics.py
  python scripts/analyze_confidence_dynamics.py --repo-root . --outputs-dir outputs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from e


def load_runs(outputs_dir: Path, repo_root: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(outputs_dir.glob("**/*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if not isinstance(d, dict) or "turns" not in d:
            continue
        md = d.get("run_metadata") or {}
        ds = md.get("dataset") or {}
        cfg = (md.get("run_config") or {}).get("cli_args") or {}
        model = cfg.get("model_id", "UNKNOWN")
        strat = d.get("strategy") or cfg.get("strategy") or "UNKNOWN"
        gold = ds.get("answer_letter")
        if gold is None and isinstance(ds.get("answer_index"), int):
            gold = chr(ord("A") + int(ds["answer_index"]))
        turns = d.get("turns") or []
        conf_seq: list[float | None] = []
        correct_seq: list[bool | None] = []
        choice_seq: list[str | None] = []
        for t in turns:
            if not isinstance(t, dict):
                conf_seq.append(None)
                correct_seq.append(None)
                choice_seq.append(None)
                continue
            ch = t.get("choice_letter")
            conf = t.get("confidence")
            conf_seq.append(float(conf) if isinstance(conf, (int, float)) else None)
            choice_seq.append(ch if isinstance(ch, str) else None)
            if ch is None or gold is None:
                correct_seq.append(None)
            else:
                correct_seq.append(ch == gold)
        try:
            rel = str(p.relative_to(repo_root))
        except ValueError:
            rel = str(p)
        rows.append(
            {
                "path": rel,
                "model": model,
                "strategy": strat,
                "question_id": d.get("question_id"),
                "final_correct": (d.get("metrics") or {}).get("final_correct"),
                "conf": conf_seq,
                "correct": correct_seq,
                "choice": choice_seq,
            }
        )
    return rows


def nanmean_stack(seqs: list[list[float | None]], max_t: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.full((len(seqs), max_t), np.nan, dtype=np.float64)
    for i, s in enumerate(seqs):
        for t, v in enumerate(s[:max_t]):
            if v is not None and isinstance(v, (int, float)):
                arr[i, t] = float(v)
    c = np.sum(~np.isnan(arr), axis=0)
    s = np.nansum(arr, axis=0)
    m = np.divide(s, c, out=np.full_like(s, np.nan), where=c > 0)
    return m, c.astype(int)


def nanstd_stack(seqs: list[list[float | None]], max_t: int) -> np.ndarray:
    arr = np.full((len(seqs), max_t), np.nan, dtype=np.float64)
    for i, s in enumerate(seqs):
        for t, v in enumerate(s[:max_t]):
            if v is not None and isinstance(v, (int, float)):
                arr[i, t] = float(v)
    c = np.sum(~np.isnan(arr), axis=0)
    m = np.divide(np.nansum(arr, axis=0), c, out=np.full((arr.shape[1],), np.nan), where=c > 0)
    var = np.divide(
        np.nansum((arr - m) ** 2, axis=0),
        (c - 1),
        out=np.full((arr.shape[1],), np.nan),
        where=c > 1,
    )
    return np.sqrt(np.maximum(var, 0.0))


def reliability_bins(confs: list[float], corrs: list[bool]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = np.zeros(10, dtype=np.float64)
    cnt = np.zeros(10, dtype=np.int64)
    for c, y in zip(confs, corrs):
        b = min(int(c // 10), 9)
        cnt[b] += 1
        acc[b] += 1.0 if y else 0.0
    with np.errstate(invalid="ignore"):
        rate = np.divide(acc, np.maximum(cnt, 1))
    centers = np.arange(5, 100, 10)
    return centers, rate, cnt


def first_flip_index(choices: list[str | None]) -> int | None:
    prev: str | None = None
    for i, ch in enumerate(choices):
        if ch is None:
            continue
        if prev is not None and ch != prev:
            return i
        prev = ch
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--outputs-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    root = args.repo_root.resolve()
    outputs_dir = (args.outputs_dir or (root / "outputs")).resolve()
    out_dir = (args.out_dir or (outputs_dir / "analysis" / "confidence_dynamics")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(outputs_dir, root)
    if not runs:
        raise SystemExit(f"No runs found under {outputs_dir}")

    max_t = max(len(r["conf"]) for r in runs)
    strategies = sorted({r["strategy"] for r in runs})
    models = sorted({r["model"] for r in runs})

    lines: list[str] = []
    lines.append("# Confidence dynamics (auto-generated)\n")
    lines.append(f"- Runs: **{len(runs)}** | Max turns: **{max_t}**\n")
    lines.append(f"- Plots directory: `{out_dir.relative_to(root)}`\n")

    # --- 1) Mean trajectory by model × strategy ---
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), squeeze=False)
    for mi, model in enumerate(models):
        ax = axes[0, mi]
        for strat in strategies:
            sub = [r["conf"] for r in runs if r["model"] == model and r["strategy"] == strat]
            if not sub:
                continue
            m, _ = nanmean_stack(sub, max_t)
            s = nanstd_stack(sub, max_t)
            t = np.arange(len(m))
            ax.plot(t, m, label=strat, marker="o", markersize=3)
            lo = np.where(np.isnan(m), np.nan, m - s)
            hi = np.where(np.isnan(m), np.nan, m + s)
            ax.fill_between(t, lo, hi, alpha=0.15)
        ax.set_title(model.split("/")[-1][:22])
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Mean confidence ± std (across runs)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Confidence trajectory: mean ± within-run spread by time")
    fig.tight_layout()
    p1 = out_dir / "trajectory_mean_std_by_model_strategy.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    lines.append(f"- ![trajectory]({p1.name}) — mean ± std per turn.\n")

    # --- 2) Per model: final correct vs wrong ---
    for model in models:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, pred in (("final_correct", True), ("final_wrong", False)):
            sub = [r["conf"] for r in runs if r["model"] == model and r["final_correct"] is pred]
            if not sub:
                continue
            m, _ = nanmean_stack(sub, max_t)
            ax.plot(np.arange(len(m)), m, marker="o", markersize=3, label=label)
        ax.set_title(f"{model.split('/')[-1]}: confidence vs turn by final outcome")
        ax.set_xlabel("Turn index")
        ax.set_ylabel("Mean confidence")
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = model.replace("/", "_").replace(" ", "_")[:60]
        pn = out_dir / f"trajectory_by_final_outcome__{safe}.png"
        fig.savefig(pn, dpi=160)
        plt.close(fig)
    lines.append("- Per-model trajectories split by **final** correctness (files `trajectory_by_final_outcome__*.png`).\n")

    # --- 3) Pooled reliability (all turns) per model × strategy ---
    fig, axes = plt.subplots(len(models), len(strategies), figsize=(4 * len(strategies), 3 * len(models)), squeeze=False)
    for mi, model in enumerate(models):
        for si, strat in enumerate(strategies):
            ax = axes[mi, si]
            confs: list[float] = []
            corrs: list[bool] = []
            for r in runs:
                if r["model"] != model or r["strategy"] != strat:
                    continue
                for c, y in zip(r["conf"], r["correct"]):
                    if c is None or y is None:
                        continue
                    confs.append(float(c))
                    corrs.append(bool(y))
            if not confs:
                ax.set_visible(False)
                continue
            centers, rate, cnt = reliability_bins(confs, corrs)
            ax.bar(centers, rate, width=8, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.plot([0, 100], [0, 1], "k--", linewidth=1, label="perfect calibration")
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence bin (center)")
            ax.set_ylabel("Empirical accuracy")
            ax.set_title(f"{model.split('/')[-1][:14]}\n{strat[:16]}\n(n={len(confs)} turns)")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Reliability: empirical accuracy vs stated confidence (pooled turns)")
    fig.tight_layout()
    p2 = out_dir / "reliability_grid_model_x_strategy.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    lines.append(f"- ![reliability]({p2.name}) — calibration-style bins (10-point wide).\n")

    # --- 4) Δ confidence on answer flip ---
    flip_deltas: list[float] = []
    flip_deltas_by_ms: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs:
        prev: str | None = None
        for i, (ch, cf) in enumerate(zip(r["choice"], r["conf"])):
            if ch is None or cf is None:
                prev = ch
                continue
            if prev is not None and ch != prev:
                prev_cf = r["conf"][i - 1]
                if prev_cf is not None:
                    d = float(cf) - float(prev_cf)
                    flip_deltas.append(d)
                    flip_deltas_by_ms[(r["model"], r["strategy"])].append(d)
            prev = ch
    fig, ax = plt.subplots(figsize=(8, 4))
    if flip_deltas:
        ax.hist(flip_deltas, bins=40, range=(-60, 60), color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Δ confidence at answer flip (vs previous turn)")
    ax.set_ylabel("Count")
    ax.set_title("Confidence change when the chosen letter flips")
    fig.tight_layout()
    p3 = out_dir / "delta_conf_on_answer_flip.png"
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    lines.append(f"- ![flip delta]({p3.name}) — how confidence moves when answers change.\n")

    # --- 5) Mean flip delta by model × strategy ---
    fig, ax = plt.subplots(figsize=(10, 4))
    keys = sorted(flip_deltas_by_ms.keys(), key=lambda k: (k[0], k[1]))
    means = [float(np.mean(flip_deltas_by_ms[k])) for k in keys]
    x = np.arange(len(keys))
    xlabs = [f"{m.split('/')[-1][:10]}\n{s[:12]}" for m, s in keys]
    ax.bar(x, means, color="#8172B3", edgecolor="white")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabs, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean Δ confidence on flip")
    ax.set_title("Average confidence jump when answer flips (by model × strategy)")
    fig.tight_layout()
    p4 = out_dir / "mean_delta_conf_on_flip_by_model_strategy.png"
    fig.savefig(p4, dpi=160)
    plt.close(fig)
    lines.append(f"- ![flip delta by pair]({p4.name})\n")

    # --- 6) Within-run confidence volatility (std of conf over valid turns) ---
    vol_by_ms: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in runs:
        vals = [float(x) for x in r["conf"] if isinstance(x, (int, float))]
        if len(vals) >= 2:
            vol_by_ms[(r["model"], r["strategy"])].append(float(np.std(vals)))
    fig, ax = plt.subplots(figsize=(10, 4))
    keys2 = sorted(vol_by_ms.keys(), key=lambda k: (k[0], k[1]))
    vm = [np.mean(vol_by_ms[k]) for k in keys2]
    x2 = np.arange(len(keys2))
    ax.bar(x2, vm, color="#CCB974", edgecolor="white")
    ax.set_xticks(x2)
    ax.set_xticklabels([f"{m.split('/')[-1][:10]}\n{s[:12]}" for m, s in keys2], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Mean within-run std(confidence)")
    ax.set_title("Confidence volatility inside each run (higher = more movement)")
    fig.tight_layout()
    p5 = out_dir / "within_run_conf_std_by_model_strategy.png"
    fig.savefig(p5, dpi=160)
    plt.close(fig)
    lines.append(f"- ![volatility]({p5.name}) — average spread of confidence inside a run.\n")

    # --- 7) Flip-aligned mean confidence (relative offset from first flip) ---
    max_off = 5
    aligned: dict[tuple[str, str], list[list[float | None]]] = defaultdict(list)
    for r in runs:
        fi = first_flip_index(r["choice"])
        if fi is None:
            continue
        seq = r["conf"]
        row: list[float | None] = []
        for off in range(-max_off, max_off + 1):
            j = fi + off
            if 0 <= j < len(seq) and seq[j] is not None:
                row.append(float(seq[j]))  # type: ignore[arg-type]
            else:
                row.append(None)
        aligned[(r["model"], r["strategy"])].append(row)
    if aligned:
        offs = list(range(-max_off, max_off + 1))
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4), squeeze=False)
        for mi, model in enumerate(models):
            ax = axes[0, mi]
            for strat in strategies:
                rows = aligned.get((model, strat), [])
                if not rows:
                    continue
                arr = np.full((len(rows), len(offs)), np.nan)
                for i, row in enumerate(rows):
                    for j, v in enumerate(row):
                        if v is not None:
                            arr[i, j] = v
                cc = np.sum(~np.isnan(arr), axis=0)
                m = np.divide(np.nansum(arr, axis=0), cc, out=np.full(arr.shape[1], np.nan), where=cc > 0)
                ax.plot(offs, m, marker="o", markersize=3, label=strat)
            ax.axvline(0, color="k", linestyle=":", linewidth=1)
            ax.set_title(model.split("/")[-1][:20])
            ax.set_xlabel("Turn offset from first answer flip")
            ax.set_ylabel("Mean confidence")
            ax.set_ylim(0, 105)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        fig.suptitle("Confidence around the first answer flip (aligned series)")
        fig.tight_layout()
        p6 = out_dir / "confidence_aligned_first_flip.png"
        fig.savefig(p6, dpi=160)
        plt.close(fig)
        lines.append(f"- ![aligned]({p6.name}) — ±5 turns around first letter change.\n")

    summary_path = out_dir / "SUMMARY.md"
    summary_path.write_text("".join(lines))
    print(f"Wrote plots and {summary_path}")


if __name__ == "__main__":
    main()
