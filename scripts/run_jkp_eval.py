#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jkp_eval.runner import run_eval


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Just Keep Prompting STAR evaluation.")
    parser.add_argument("--config", required=True, help="Path to config JSON file.")
    parser.add_argument("--limit", type=int, default=None, help="Override dataset limit.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompt previews without loading any model.",
    )
    args = parser.parse_args()

    artifacts = run_eval(args.config, limit=args.limit, dry_run=args.dry_run)
    if args.dry_run:
        print(json.dumps(artifacts.records, indent=2))
    else:
        print(f"Wrote {len(artifacts.records)} records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

