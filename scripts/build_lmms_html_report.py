#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _safe(v, default="-"):
    return default if v is None else v


def _format_text_block(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br>")


def build_report(samples_path: Path, results_path: Path, dataset_path: Path, output_path: Path, title: str):
    results = _read_json(results_path)
    samples = _read_jsonl(samples_path)
    dataset_rows = _read_jsonl(dataset_path)

    video_by_question = {row.get("question_id"): row.get("video_path") for row in dataset_rows}

    task_name = "jkp_star_negation"
    task_metrics = results.get("results", {}).get(task_name, {})
    throughput = results.get("throughput", {})
    config = results.get("config", {})
    task_config = results.get("configs", {}).get(task_name, {})

    cards = []
    for idx, sample in enumerate(samples, start=1):
        try:
            payload = json.loads(sample.get("filtered_resps", "{}"))
        except json.JSONDecodeError:
            payload = {}

        question_id = payload.get("question_id", f"sample_{idx}")
        video_path = video_by_question.get(question_id, "")
        trace = payload.get("trace", [])
        rounds = payload.get("agentic_rounds", [])
        metrics = payload.get("metrics", {})
        target = sample.get("target")

        rounds_html = []
        for r in rounds:
            rounds_html.append(
                f"""
                <details class="round">
                  <summary>Round {_safe(r.get("round_idx"))} {'(terminal)' if r.get("terminal") else ''}</summary>
                  <div class="grid">
                    <div>
                      <h5>Input</h5>
                      <div class="mono">{_format_text_block(str(r.get("round_input", "")))}</div>
                    </div>
                    <div>
                      <h5>Model Output</h5>
                      <div class="mono">{_format_text_block(str(r.get("model_output", "")))}</div>
                    </div>
                  </div>
                </details>
                """
            )

        trace_rows = []
        for t_idx, t in enumerate(trace):
            if isinstance(t, dict):
                turn_index = _safe(t.get("turn_index"), t_idx)
                choice_letter = _safe(t.get("choice_letter"))
                confidence = _safe(t.get("confidence"))
                raw_text = str(t.get("raw_text", ""))
            else:
                turn_index = t_idx
                choice_letter = "-"
                confidence = "-"
                raw_text = str(t)
            trace_rows.append(
                f"<tr><td>{turn_index}</td><td>{html.escape(str(choice_letter))}</td><td>{confidence}</td><td>{_format_text_block(raw_text)}</td></tr>"
            )

        video_html = ""
        if video_path:
            escaped_video = html.escape(video_path, quote=True)
            video_html = f"""
            <div class="video-block">
              <div><strong>Video:</strong> <code>{escaped_video}</code></div>
              <video controls preload="metadata" src="{escaped_video}"></video>
            </div>
            """

        cards.append(
            f"""
            <section class="card">
              <h3>#{idx}: {html.escape(str(question_id))}</h3>
              <div class="meta">
                <span><strong>Target:</strong> {html.escape(str(target))}</span>
                <span><strong>Initial acc:</strong> {_safe(sample.get('jkp_initial_acc'))}</span>
                <span><strong>Final acc:</strong> {_safe(sample.get('jkp_final_acc'))}</span>
                <span><strong>Flip rate:</strong> {_safe(sample.get('jkp_flip_rate'))}</span>
              </div>
              {video_html}
              <details open>
                <summary>Per-turn trace</summary>
                <table>
                  <thead><tr><th>Turn</th><th>Choice</th><th>Conf</th><th>Raw Text</th></tr></thead>
                  <tbody>
                    {''.join(trace_rows)}
                  </tbody>
                </table>
              </details>
              <details>
                <summary>Agentic rounds (input/output)</summary>
                {''.join(rounds_html)}
              </details>
              <details>
                <summary>Sample metrics JSON</summary>
                <pre>{html.escape(json.dumps(metrics, indent=2))}</pre>
              </details>
            </section>
            """
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1, h2, h3 {{ margin: 0 0 12px 0; }}
    .summary {{ background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 16px; margin-bottom: 20px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
    .metric {{ background: #1f2937; border-radius: 8px; padding: 10px; }}
    .card {{ background: #111827; border: 1px solid #374151; border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 10px; }}
    .video-block {{ margin: 10px 0 14px; }}
    video {{ width: 100%; max-height: 420px; background: #000; border-radius: 8px; margin-top: 8px; }}
    details {{ margin: 8px 0; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #374151; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #1f2937; }}
    .grid {{ display: grid; gap: 10px; grid-template-columns: 1fr 1fr; }}
    .mono, pre, code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .mono, pre {{ background: #020617; border: 1px solid #334155; border-radius: 8px; padding: 10px; overflow-x: auto; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{html.escape(title)}</h1>
    <div class="summary">
      <h2>Run Summary</h2>
      <div class="metrics">
        <div class="metric"><strong>Model</strong><br>{html.escape(str(config.get('model_args', '')))}</div>
        <div class="metric"><strong>Limit</strong><br>{html.escape(str(config.get('limit', '-')))}</div>
        <div class="metric"><strong>Final Acc</strong><br>{html.escape(str(task_metrics.get('jkp_final_acc,none', '-')))}</div>
        <div class="metric"><strong>Initial Acc</strong><br>{html.escape(str(task_metrics.get('jkp_initial_acc,none', '-')))}</div>
        <div class="metric"><strong>Flip Rate</strong><br>{html.escape(str(task_metrics.get('jkp_flip_rate,none', '-')))}</div>
        <div class="metric"><strong>Total Tokens</strong><br>{html.escape(str(throughput.get('total_gen_tokens', '-')))}</div>
        <div class="metric"><strong>Total Gen Time (s)</strong><br>{html.escape(str(round(throughput.get('total_elapsed_time', 0), 2)))}</div>
        <div class="metric"><strong>Avg Speed (tok/s)</strong><br>{html.escape(str(round(throughput.get('avg_speed', 0), 3)))}</div>
      </div>
      <details>
        <summary>Full Run Configuration (from results.json)</summary>
        <pre>{html.escape(json.dumps(config, indent=2))}</pre>
      </details>
      <details>
        <summary>Full Task Configuration (resolved task yaml)</summary>
        <pre>{html.escape(json.dumps(task_config, indent=2))}</pre>
      </details>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""

    output_path.write_text(doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build an HTML report from LMMS JSONL samples + results.")
    parser.add_argument("--samples", required=True, type=Path, help="Path to *_samples_*.jsonl")
    parser.add_argument("--results", required=True, type=Path, help="Path to *_results.json")
    parser.add_argument("--dataset", required=True, type=Path, help="Path to task dataset jsonl for video paths")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML file path")
    parser.add_argument("--title", default="LMMS Eval Report")
    args = parser.parse_args()

    build_report(args.samples, args.results, args.dataset, args.output, args.title)
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
