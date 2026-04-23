from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class BackendResponse:
    text: str


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


class TransformersMultimodalBackend:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.family = config["family"]
        self.model_id = config["model_id"]
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.temperature = float(config.get("temperature", 0.2))
        self.attn_implementation = config.get("attn_implementation")

        from transformers import AutoProcessor

        quantization_config = None
        quantization = config.get("quantization")
        if quantization in {"4bit", "8bit"}:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
            )

        common_kwargs = {
            "device_map": config.get("device_map", "auto"),
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
            "low_cpu_mem_usage": True,
        }
        if quantization_config is not None:
            common_kwargs["quantization_config"] = quantization_config
        else:
            common_kwargs["torch_dtype"] = _torch_dtype(config.get("dtype", "float16"))
        if self.attn_implementation:
            common_kwargs["attn_implementation"] = self.attn_implementation

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        if self.family == "qwen3_vl":
            from transformers import Qwen3VLMoeForConditionalGeneration

            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                **common_kwargs,
            )
        elif self.family == "internvl3_5":
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported model family: {self.family}")

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        frame_paths: list[Path] | None = None,
    ) -> BackendResponse:
        if self.family == "qwen3_vl":
            return self._generate_qwen(messages, frame_paths=frame_paths)
        return self._generate_generic(messages)

    def _generate_generic(self, messages: list[dict[str, Any]]) -> BackendResponse:
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generated = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        input_len = inputs["input_ids"].shape[1]
        output_text = self.processor.decode(
            generated[0, input_len:],
            skip_special_tokens=True,
        )
        return BackendResponse(text=output_text)

    def _generate_qwen(
        self,
        messages: list[dict[str, Any]],
        *,
        frame_paths: list[Path] | None = None,
    ) -> BackendResponse:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise RuntimeError(
                "qwen-vl-utils is required for Qwen3-VL. Run scripts/setup_vlm_env.sh first."
            ) from exc

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)
        generated_ids = self.model.generate(
            **inputs,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return BackendResponse(text=output_text)


class TransformersTextBackend:
    def __init__(self, config: dict[str, Any]):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
        kwargs = {
            "device_map": config.get("device_map", "auto"),
            "low_cpu_mem_usage": True,
        }
        quantization = config.get("quantization")
        if quantization in {"4bit", "8bit"}:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=quantization == "4bit",
                load_in_8bit=quantization == "8bit",
            )
        else:
            kwargs["torch_dtype"] = _torch_dtype(config.get("dtype", "float16"))
        self.model = AutoModelForCausalLM.from_pretrained(config["model_id"], **kwargs)

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(
            **inputs,
            do_sample=self.config.get("temperature", 0.0) > 0,
            temperature=float(self.config.get("temperature", 0.0)),
            max_new_tokens=int(self.config.get("max_new_tokens", 96)),
        )
        generated = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())

