#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.metadata
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jkp_infer.backends import (
    GeminiNativeChatBackend,
    OpenAICompatibleChatBackend,
    TransformersImageChatBackend,
)
from jkp_infer.dataset import load_star_examples
from jkp_infer.pipeline import run_example_chat, write_run_artifact


def _select_examples(
    *,
    qa_path: Path,
    metadata_path: Path,
    categories: list[str] | None,
    limit: int | None,
    per_category_limit: int | None,
):
    examples = load_star_examples(
        qa_path,
        metadata_path,
        categories=categories,
        limit=None,
    )
    if per_category_limit is not None:
        selected = []
        seen_per_category: dict[str, int] = defaultdict(int)
        for example in examples:
            category_key = example.category.lower()
            if seen_per_category[category_key] >= per_category_limit:
                continue
            selected.append(example)
            seen_per_category[category_key] += 1
        examples = selected
    if limit is not None:
        examples = examples[:limit]
    return examples


def _examples_from_question_ids_file(
    *,
    qa_path: Path,
    metadata_path: Path,
    categories: list[str] | None,
    question_ids_path: Path,
):
    wanted: list[str] = []
    for line in question_ids_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            wanted.append(line)
    if not wanted:
        raise SystemExit(f"No question ids found in {question_ids_path}")
    examples = load_star_examples(
        qa_path,
        metadata_path,
        categories=categories,
        limit=None,
    )
    by_id = {ex.question_id: ex for ex in examples}
    missing = [q for q in wanted if q not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} question_id(s) not in dataset (showing first 10): {missing[:10]}"
        )
    return [by_id[q] for q in wanted]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a self-hosted JKP smoke test without vendor dependencies.")
    parser.add_argument(
        "--backend",
        choices=["transformers", "openai_compatible", "gemini_native"],
        default="gemini_native",
        help="Use local transformers backend, OpenAI-compatible API backend, or Gemini native backend.",
    )
    parser.add_argument(
        "--family",
        choices=["qwen3_vl", "internvl3_5", "gemini"],
        default="qwen3_vl",
        help="Model family for --backend transformers.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="HF model id for --backend transformers.",
    )
    parser.add_argument(
        "--model-preset",
        choices=["qwen3_vl_30b_a3b", "internvl3_5_38b", "internvl3_5_30b_a3b", "gemini_3_1_pro_preview"],
        default=None,
        help=(
            "Optional named model preset. If set, it determines --family and default --model-id "
            "(unless --model-id is also provided)."
        ),
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument(
        "--max-memory-per-gpu-gib",
        type=float,
        default=None,
        help=(
            "Optional per-visible-GPU memory cap (GiB) used for model sharding "
            "(e.g. 28). Helps reduce single-GPU hotspot OOMs."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Comma-separated GPU ids to expose to this process (example: 0,1,2,3).",
    )
    parser.add_argument(
        "--skip-v100-validation",
        action="store_true",
        help="When using --cuda-visible-devices, do not require every visible GPU to be a V100.",
    )
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", dest="low_cpu_mem_usage", action="store_true", default=True)
    parser.add_argument("--no-low-cpu-mem-usage", dest="low_cpu_mem_usage", action="store_false")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument(
        "--api-base-url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        help="Base URL for --backend openai_compatible.",
    )
    parser.add_argument(
        "--api-key-env-var",
        default="GEMINI_API_KEY",
        help="Env var holding the API key for --backend openai_compatible.",
    )
    parser.add_argument(
        "--min-seconds-between-requests",
        type=float,
        default=4.0,
        help="Throttle requests for --backend openai_compatible to avoid Tier 1 rate-limit overruns.",
    )
    parser.add_argument(
        "--max-api-requests",
        type=int,
        default=900,
        help=(
            "Hard cap on total API calls in this run for --backend openai_compatible. "
            "Set <= your Tier 1 quota budget."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--per-category-limit",
        type=int,
        default=None,
        help="Select up to this many examples from each category.",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Optional category filter (e.g. --categories Feasibility Interaction Sequence). Defaults to all.",
    )
    parser.add_argument(
        "--strategy",
        choices=["adversarial_negation", "pure_socratic", "context_socratic"],
        default="adversarial_negation",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument(
        "--input-mode",
        choices=["frames", "video"],
        default="frames",
        help="Use extracted frame images or pass the full video directly to the model.",
    )
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Sample frames across the full video at this FPS (e.g. 1.0 = 1 frame/sec).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap extracted frames when --fps is used.",
    )
    parser.add_argument("--output-dir", default="outputs/self_hosted_smoke")
    parser.add_argument(
        "--text-summary-model-id",
        default="Qwen/Qwen3-8B",
        help="Dense LM for context_socratic rationale summarization (ignored with --skip-text-summarizer).",
    )
    parser.add_argument(
        "--text-summary-dtype",
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="dtype for the text summarizer weights.",
    )
    parser.add_argument(
        "--text-summary-device-map",
        default="auto",
        help='device_map for the text summarizer (default: auto). Use "cpu" only if loading alongside the VL model OOMs.',
    )
    parser.add_argument(
        "--skip-text-summarizer",
        action="store_true",
        help="For context_socratic, use heuristic rationale truncation instead of loading Qwen3 text.",
    )
    parser.add_argument(
        "--question-ids-file",
        default=None,
        help=(
            "Path to a text file: one question_id per line. Run those examples in that order; "
            "ignores --limit and --per-category-limit."
        ),
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Send only text prompts (no video/frames) to debug output formatting cheaply.",
    )
    parser.add_argument(
        "--debug-gemini-io",
        action="store_true",
        help="Print Gemini raw response debug info (finish reason, token usage, raw text).",
    )
    parser.add_argument(
        "--print-raw-turns",
        action="store_true",
        help="Print full raw model text for each turn to stdout.",
    )
    args = parser.parse_args()

    if args.fps is not None and args.fps <= 0:
        raise SystemExit("--fps must be > 0.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames must be > 0.")
    if args.per_category_limit is not None and args.per_category_limit <= 0:
        raise SystemExit("--per-category-limit must be > 0.")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    qa_path = ROOT / "star_data/star_clips_qa.json"
    metadata_path = ROOT / "star_data/star_clips_metadata.json"
    if args.question_ids_file:
        if args.limit is not None or args.per_category_limit is not None:
            raise SystemExit("Do not use --limit or --per-category-limit with --question-ids-file.")
        examples = _examples_from_question_ids_file(
            qa_path=qa_path,
            metadata_path=metadata_path,
            categories=args.categories,
            question_ids_path=Path(args.question_ids_file).expanduser().resolve(),
        )
    else:
        examples = _select_examples(
            qa_path=qa_path,
            metadata_path=metadata_path,
            categories=args.categories,
            limit=args.limit,
            per_category_limit=args.per_category_limit,
        )
    if not examples:
        raise SystemExit("No STAR examples loaded.")

    if args.model_preset:
        preset_to_family_model = {
            "qwen3_vl_30b_a3b": ("qwen3_vl", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
            "internvl3_5_38b": ("internvl3_5", "OpenGVLab/InternVL3_5-38B-HF"),
            "internvl3_5_30b_a3b": ("internvl3_5", "OpenGVLab/InternVL3_5-30B-A3B-HF"),
            "gemini_3_1_pro_preview": ("gemini", "gemini-3.1-pro-preview"),
        }
        preset_family, preset_model_id = preset_to_family_model[args.model_preset]
        args.family = preset_family
        if not args.model_id:
            args.model_id = preset_model_id

    if args.backend == "transformers":
        import torch

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "<all>")
        print(f"CUDA_VISIBLE_DEVICES={visible_devices}")
        print(f"torch.cuda.device_count()={torch.cuda.device_count()}")
        gpu_inventory: list[dict[str, object]] = []
        for gpu_index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(gpu_index)
            total_gib = torch.cuda.get_device_properties(gpu_index).total_memory / (1024**3)
            print(f"  cuda:{gpu_index} name={name} total_mem={total_gib:.1f}GiB")
            gpu_inventory.append({"index": gpu_index, "name": name, "total_memory_gib": round(total_gib, 2)})
        if args.cuda_visible_devices:
            requested = [gpu.strip() for gpu in args.cuda_visible_devices.split(",") if gpu.strip()]
            detected = torch.cuda.device_count()
            if detected != len(requested):
                raise SystemExit(
                    f"Expected {len(requested)} visible GPUs from --cuda-visible-devices, but detected {detected}."
                )
            if not args.skip_v100_validation:
                for gpu_index in range(detected):
                    if "V100" not in torch.cuda.get_device_name(gpu_index).upper():
                        raise SystemExit(
                            f"cuda:{gpu_index} is not a V100 ({torch.cuda.get_device_name(gpu_index)}). "
                            "Pass --skip-v100-validation to run on other GPUs."
                        )

        default_model_ids = {
            "qwen3_vl": "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "internvl3_5": "OpenGVLab/InternVL3_5-38B-HF",
        }
        max_memory = None
        if args.max_memory_per_gpu_gib is not None:
            max_mem = f"{int(args.max_memory_per_gpu_gib)}GiB"
            max_memory = {gpu_index: max_mem for gpu_index in range(torch.cuda.device_count())}
        backend = TransformersImageChatBackend(
            {
                "family": args.family,
                "model_id": args.model_id or default_model_ids[args.family],
                "dtype": args.dtype,
                "device_map": args.device_map,
                "load_in_8bit": args.load_in_8bit,
                "load_in_4bit": args.load_in_4bit,
                "low_cpu_mem_usage": args.low_cpu_mem_usage,
                "attn_implementation": args.attn_implementation,
                "cache_dir": args.cache_dir,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "trust_remote_code": args.trust_remote_code,
                "max_memory": max_memory,
            }
        )
    elif args.backend == "openai_compatible":
        estimated_requests = len(examples) * (args.max_turns + 1)
        if estimated_requests > args.max_api_requests:
            raise SystemExit(
                "Estimated API requests would exceed --max-api-requests "
                f"({estimated_requests} > {args.max_api_requests}). "
                "Increase the cap or reduce examples/turns."
            )
        default_model_ids = {
            "gemini": "gemini-3.1-pro-preview",
        }
        backend = OpenAICompatibleChatBackend(
            {
                "model_id": args.model_id or default_model_ids.get(args.family, "gemini-3.1-pro-preview"),
                "api_base_url": args.api_base_url,
                "api_key_env_var": args.api_key_env_var,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "min_seconds_between_requests": args.min_seconds_between_requests,
            }
        )
        gpu_inventory = []
    else:
        estimated_requests = len(examples) * (args.max_turns + 1)
        if estimated_requests > args.max_api_requests:
            raise SystemExit(
                "Estimated API requests would exceed --max-api-requests "
                f"({estimated_requests} > {args.max_api_requests}). "
                "Increase the cap or reduce examples/turns."
            )
        default_model_ids = {
            "gemini": "gemini-3.1-pro-preview",
        }
        backend = GeminiNativeChatBackend(
            {
                "model_id": args.model_id or default_model_ids.get(args.family, "gemini-3.1-pro-preview"),
                "api_key_env_var": args.api_key_env_var,
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
                "min_seconds_between_requests": args.min_seconds_between_requests,
                "debug_io": args.debug_gemini_io,
            }
        )
        gpu_inventory = []
    try:
        qwen_vl_utils_version = importlib.metadata.version("qwen-vl-utils")
    except importlib.metadata.PackageNotFoundError:
        qwen_vl_utils_version = None
    try:
        google_genai_version = importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        google_genai_version = None

    run_config = {
        "cli_args": vars(args),
        "libraries": {
            "torch": importlib.metadata.version("torch"),
            "transformers": importlib.metadata.version("transformers"),
            "qwen_vl_utils": qwen_vl_utils_version,
            "openai": importlib.metadata.version("openai"),
            "google_genai": google_genai_version,
        },
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpus": gpu_inventory,
        },
    }
    output_dir = ROOT / args.output_dir
    def _text_summarizer_factory():
        if args.strategy != "context_socratic" or args.skip_text_summarizer:
            return None
        from jkp_infer.qwen_text_summarize import QwenTextSummarizer

        print(
            f"Loading text summarizer for context_socratic: {args.text_summary_model_id} "
            f"(dtype={args.text_summary_dtype}, device_map={args.text_summary_device_map})",
            flush=True,
        )
        return QwenTextSummarizer(
            {
                "model_id": args.text_summary_model_id,
                "dtype": args.text_summary_dtype,
                "device_map": args.text_summary_device_map,
                "max_new_tokens": 96,
                "temperature": 0.0,
            }
        )

    text_summarizer = _text_summarizer_factory()

    print(
        f"Running smoke test for {len(examples)} example(s) "
        f"with backend={args.backend} strategy={args.strategy}"
    )

    for example in examples:
        run = run_example_chat(
            example=example,
            strategy=args.strategy,
            backend=backend,
            max_turns=args.max_turns,
            frame_count=args.frame_count if args.fps is None else None,
            frame_fps=args.fps,
            frame_max=args.max_frames,
            input_mode=args.input_mode,
            run_config=run_config,
            output_dir=output_dir,
            text_summarizer=text_summarizer,
            include_visual=not args.text_only,
        )
        artifact_path = write_run_artifact(
            run,
            output_dir / "runs" / f"{example.question_id}_{args.strategy}.json",
        )
        print(f"\nExample: {example.question_id}")
        print(f"Artifact: {artifact_path}")
        print(f"Metrics: {run.metrics}")
        print("Answer trace:")
        for turn in run.turns:
            print(
                f"  turn={turn['turn_index']} answer={turn['choice_letter']} "
                f"confidence={turn['confidence']}"
            )
            if args.print_raw_turns:
                print(f"  raw_text[{turn['turn_index']}]:\n{turn['raw_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

