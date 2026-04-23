from __future__ import annotations

import os
import re
from typing import Any

_SUMMARY_SYSTEM = (
    "You compress another model's multiple-choice reply into a single short sentence for a follow-up prompt. "
    "Be faithful to the text; do not invent details."
)


def _first_sentence(text: str) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return ""
    match = re.match(r"([^.!?]+[.!?]?)", cleaned)
    return (match.group(1) if match else cleaned).strip()


class QwenTextSummarizer:
    """Dense causal LM (e.g. Qwen/Qwen3-8B) for one-sentence rationale summaries."""

    def __init__(self, config: dict[str, Any]):
        self.model_id = str(config["model_id"])
        self.device_map = config.get("device_map", "auto")
        self.max_new_tokens = int(config.get("max_new_tokens", 96))
        self.temperature = float(config.get("temperature", 0.0))
        self.do_sample = bool(config.get("do_sample", self.temperature > 0))
        self.hf_token = config.get("hf_token") or os.getenv("HF_TOKEN")
        self.cache_dir = config.get("cache_dir")
        self.trust_remote_code = bool(config.get("trust_remote_code", True))
        self.attn_implementation = config.get("attn_implementation")

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        # float16 is the safer default on many NVIDIA GPUs (bf16 can raise CUBLAS_STATUS_NOT_SUPPORTED).
        dtype_name = str(config.get("dtype", "float16"))
        torch_dtype = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }.get(dtype_name, torch.float16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "token": self.hf_token,
            "cache_dir": self.cache_dir,
            "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
        }
        if self.attn_implementation:
            model_kwargs["attn_implementation"] = self.attn_implementation
        try:
            model_kwargs["dtype"] = torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        except TypeError:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        self.model.eval()

    def summarize_rationale_one_sentence(self, rationale_text: str) -> str:
        user = (
            "Summarize in exactly one sentence the following rationale text. "
            "Preserve the core reasoning; do not add labels like ANSWER or RATIONALE.\n\n"
            f"{rationale_text.strip()}"
        )
        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM},
            {"role": "user", "content": user},
        ]
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        from torch import inference_mode

        param_device = next(self.model.parameters()).device
        inputs = inputs.to(param_device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.do_sample:
            if self.temperature > 0:
                generate_kwargs["temperature"] = self.temperature

        with inference_mode():
            out = self.model.generate(**inputs, **generate_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        decoded = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
        return _first_sentence(decoded)

    def __call__(self, rationale_text: str) -> str:
        return self.summarize_rationale_one_sentence(rationale_text)


_env_summarizer: QwenTextSummarizer | None = None
_env_summarizer_key: str | None = None


def load_env_cached_summarizer() -> QwenTextSummarizer:
    """Load (or return cached) summarizer using ``JKP_QWEN_TEXT_SUMMARY_MODEL``."""
    global _env_summarizer, _env_summarizer_key
    model_id = os.environ.get("JKP_QWEN_TEXT_SUMMARY_MODEL", "").strip()
    if not model_id:
        raise RuntimeError("JKP_QWEN_TEXT_SUMMARY_MODEL is not set.")
    dtype = os.environ.get("JKP_QWEN_TEXT_SUMMARY_DTYPE", "float16").strip() or "float16"
    device_map = os.environ.get("JKP_QWEN_TEXT_SUMMARY_DEVICE_MAP", "auto").strip() or "auto"
    cache_key = f"{model_id}|{dtype}|{device_map}"
    if _env_summarizer is None or _env_summarizer_key != cache_key:
        _env_summarizer = QwenTextSummarizer(
            {"model_id": model_id, "dtype": dtype, "device_map": device_map}
        )
        _env_summarizer_key = cache_key
    return _env_summarizer
