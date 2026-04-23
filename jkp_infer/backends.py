from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import base64
import os
from pathlib import Path
import re
import time
from typing import Any, Protocol

import cv2
from .conversation import ConversationState


@dataclass
class BackendResponse:
    text: str
    token_usage: dict[str, int] | None = None


class ChatBackend(Protocol):
    def generate(self, state: ConversationState, turn_index: int) -> BackendResponse:
        ...


class TransformersImageChatBackend:
    """Minimal image-chat backend for future real-model runs."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_id = config["model_id"]
        self.family = config["family"]
        self.temperature = float(config.get("temperature", 0.2))
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.attn_implementation = config.get("attn_implementation")
        self.do_sample = bool(config.get("do_sample", self.temperature > 0))
        self.top_p = config.get("top_p")
        self.top_k = config.get("top_k")
        self.hf_token = config.get("hf_token") or os.getenv("HF_TOKEN")
        self.cache_dir = config.get("cache_dir")
        self._qwen_vision_cache: dict[str, dict[str, Any]] = {}

        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=bool(config.get("trust_remote_code", True)),
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )

        common_kwargs = {
            "device_map": config.get("device_map", "auto"),
            "low_cpu_mem_usage": bool(config.get("low_cpu_mem_usage", True)),
            "trust_remote_code": bool(config.get("trust_remote_code", True)),
            "token": self.hf_token,
            "cache_dir": self.cache_dir,
        }
        if "max_memory" in config:
            common_kwargs["max_memory"] = config["max_memory"]
        if "offload_folder" in config:
            common_kwargs["offload_folder"] = config["offload_folder"]
        if self.attn_implementation:
            common_kwargs["attn_implementation"] = self.attn_implementation

        import torch

        if bool(config.get("load_in_8bit", False)) or bool(config.get("load_in_4bit", False)):
            from transformers import BitsAndBytesConfig

            common_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=bool(config.get("load_in_8bit", False)),
                load_in_4bit=bool(config.get("load_in_4bit", False)),
            )
        else:
            dtype_name = config.get("dtype", "float16")
            common_kwargs["torch_dtype"] = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }[dtype_name]

        if self.family == "internvl3_5":
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **common_kwargs,
            )
        elif self.family == "qwen3_vl":
            from transformers import Qwen3VLMoeForConditionalGeneration

            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                self.model_id,
                **common_kwargs,
            )
        else:
            raise ValueError(f"Unsupported family: {self.family}")

    def generate(self, state: ConversationState, turn_index: int) -> BackendResponse:
        if self.family == "qwen3_vl":
            qwen_messages = _format_qwen_messages(state.messages)
            prompt_text = self.processor.apply_chat_template(
                qwen_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            cache_key = _qwen_state_cache_key(state)
            cached_vision = self._qwen_vision_cache.get(cache_key)
            if cached_vision is None:
                from qwen_vl_utils import process_vision_info

                try:
                    image_inputs, video_inputs, processed_video_kwargs = process_vision_info(
                        qwen_messages,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                except TypeError:
                    # Backward compatibility for older qwen-vl-utils versions.
                    image_inputs, video_inputs = process_vision_info(qwen_messages)
                    processed_video_kwargs = {}
                video_metadata = None
                if video_inputs is not None:
                    # qwen-vl-utils can return [(video_tensor, metadata), ...]
                    # when return_video_metadata=True.
                    if video_inputs and isinstance(video_inputs[0], tuple):
                        video_inputs, video_metadata = map(list, zip(*video_inputs))
                cached_vision = {
                    "image_inputs": image_inputs,
                    "video_inputs": video_inputs,
                    "video_metadata": video_metadata,
                    "processed_video_kwargs": processed_video_kwargs,
                }
                self._qwen_vision_cache[cache_key] = cached_vision

            inputs = self.processor(
                text=[prompt_text],
                images=cached_vision["image_inputs"],
                videos=cached_vision["video_inputs"],
                video_metadata=cached_vision["video_metadata"],
                **cached_vision["processed_video_kwargs"],
                return_tensors="pt",
            )
        else:
            internvl_messages, image_inputs = _prepare_internvl_messages_and_images(state.messages)
            prompt_text = self.processor.apply_chat_template(
                internvl_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                text=[prompt_text],
                images=image_inputs if image_inputs else None,
                return_tensors="pt",
            )

        inputs = inputs.to(self.model.device)
        generate_kwargs: dict[str, Any] = {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
        }
        if self.do_sample and self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            generate_kwargs["top_p"] = float(self.top_p)
        if self.top_k is not None:
            generate_kwargs["top_k"] = int(self.top_k)
        generated = self.model.generate(
            **inputs,
            **generate_kwargs,
        )
        prompt_len = inputs["input_ids"].shape[1]
        output_text = self.processor.decode(
            generated[0, prompt_len:],
            skip_special_tokens=True,
        )
        completion_tokens = int(generated.shape[1] - prompt_len)
        token_usage = {
            "prompt_tokens": int(prompt_len),
            "completion_tokens": completion_tokens,
            "total_tokens": int(prompt_len + completion_tokens),
        }
        return BackendResponse(text=output_text, token_usage=token_usage)


class OpenAICompatibleChatBackend:
    """Chat backend for OpenAI-compatible multimodal APIs (including Gemini endpoints)."""

    def __init__(self, config: dict[str, Any]):
        from openai import OpenAI

        self.config = config
        self.model_id = str(config["model_id"])
        self.temperature = float(config.get("temperature", 0.2))
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.image_detail = str(config.get("image_detail", "low"))
        self.image_max_side = int(config.get("image_max_side", 512))
        self.image_jpeg_quality = int(config.get("image_jpeg_quality", 60))
        self.image_max_size_mb = float(config.get("image_max_size_mb", 20.0))
        self.image_resize_factor = float(config.get("image_resize_factor", 0.75))
        self.image_min_side = int(config.get("image_min_side", 100))
        self.debug_io = bool(config.get("debug_io", False))
        self.max_tokens_per_minute = int(config.get("max_tokens_per_minute", 30000))
        self.max_requests_per_minute = int(config.get("max_requests_per_minute", 500))
        self.max_retries = int(config.get("max_retries", 5))
        self.retry_backoff_seconds = float(config.get("retry_backoff_seconds", 8.0))
        self.api_base_url = config.get("api_base_url")
        self.api_key = (
            config.get("api_key")
            or os.getenv(str(config.get("api_key_env_var", "GEMINI_API_KEY")))
            or os.getenv("OPENAI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Missing API key. Set GEMINI_API_KEY (or OPENAI_API_KEY), "
                "or pass api_key in backend config."
            )
        self.timeout_seconds = float(config.get("timeout_seconds", 120.0))
        self.min_seconds_between_requests = float(config.get("min_seconds_between_requests", 0.0))
        self._last_request_at = 0.0
        self._request_timestamps: deque[float] = deque()
        self._recent_total_tokens: deque[tuple[float, int]] = deque()
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base_url,
            timeout=self.timeout_seconds,
        )

    def generate(self, state: ConversationState, turn_index: int) -> BackendResponse:
        if self.min_seconds_between_requests > 0:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.min_seconds_between_requests - elapsed
            if remaining > 0:
                time.sleep(remaining)

        messages = _format_openai_compatible_messages(
            state.messages,
            image_detail=self.image_detail,
            image_max_side=self.image_max_side,
            image_jpeg_quality=self.image_jpeg_quality,
            image_max_size_mb=self.image_max_size_mb,
            image_resize_factor=self.image_resize_factor,
            image_min_side=self.image_min_side,
        )
        response = None
        for attempt in range(self.max_retries):
            expected_tokens = self._expected_tokens_for_next_request(messages)
            self._wait_for_rate_budget(expected_tokens=expected_tokens)
            if self.debug_io:
                self._debug_request(
                    turn_index=turn_index,
                    messages=messages,
                    expected_tokens=expected_tokens,
                    attempt=attempt + 1,
                )
            try:
                response = self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )
                break
            except Exception as exc:
                err_text = str(exc)
                reduced = self._reduce_images_on_tpm_error(messages, err_text)
                if self.debug_io:
                    print(
                        f"[OpenAI debug:error] turn={turn_index} attempt={attempt + 1} reduced_images={reduced} error={err_text}",
                        flush=True,
                    )
                if attempt == self.max_retries - 1:
                    raise
                sleep_s = self.retry_backoff_seconds * (2**attempt)
                time.sleep(min(sleep_s, 60.0))
        if response is None:  # pragma: no cover
            raise RuntimeError("OpenAI request failed without response.")
        self._last_request_at = time.monotonic()
        self._record_request_timestamp()

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        token_usage = None
        if usage is not None:
            token_usage = {
                "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            self._record_total_tokens(token_usage["total_tokens"])
        if self.debug_io:
            self._debug_response(turn_index=turn_index, token_usage=token_usage, text=text)
        return BackendResponse(text=text, token_usage=token_usage)

    def _debug_request(
        self,
        *,
        turn_index: int,
        messages: list[dict[str, Any]],
        expected_tokens: int,
        attempt: int,
    ) -> None:
        image_parts = 0
        text_parts = 0
        approx_chars = 0
        role_counts: dict[str, int] = {}
        for msg in messages:
            role = str(msg.get("role", "unknown"))
            role_counts[role] = role_counts.get(role, 0) + 1
            content = msg.get("content", "")
            if isinstance(content, str):
                approx_chars += len(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts += 1
                    approx_chars += len(str(block.get("text", "")))
                elif block_type == "image_url":
                    image_parts += 1
                    image_url = block.get("image_url", {})
                    if isinstance(image_url, dict):
                        approx_chars += len(str(image_url.get("url", "")))
        print(
            (
                f"[OpenAI debug:req] turn={turn_index} msgs={len(messages)} roles={role_counts} "
                f"text_parts={text_parts} image_parts={image_parts} approx_chars={approx_chars} "
                f"detail={self.image_detail} max_side={self.image_max_side} jpeg_quality={self.image_jpeg_quality} "
                f"expected_tokens={expected_tokens} attempt={attempt}"
            ),
            flush=True,
        )

    def _debug_response(self, *, turn_index: int, token_usage: dict[str, int] | None, text: str) -> None:
        print(
            f"[OpenAI debug:resp] turn={turn_index} token_usage={token_usage} text_preview={repr(text[:220])}",
            flush=True,
        )

    def _expected_tokens_for_next_request(self, messages: list[dict[str, Any]]) -> int:
        # Use recent real usage when available, with a cushion.
        self._prune_windows(time.monotonic())
        if self._recent_total_tokens:
            avg_recent = sum(tokens for _, tokens in self._recent_total_tokens) / len(self._recent_total_tokens)
            return int(max(1000, avg_recent * 1.15))
        # Cold-start heuristic: low-detail images are usually lower than high-detail tiles.
        image_parts = 0
        text_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                text_chars += len(content)
                continue
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image_url":
                    image_parts += 1
                elif block.get("type") == "text":
                    text_chars += len(str(block.get("text", "")))
        return int(max(1000, image_parts * 90 + text_chars / 4 + self.max_new_tokens))

    def _wait_for_rate_budget(self, *, expected_tokens: int) -> None:
        while True:
            now = time.monotonic()
            self._prune_windows(now)
            wait_s = 0.0
            if self.max_requests_per_minute > 0 and len(self._request_timestamps) >= self.max_requests_per_minute:
                wait_s = max(wait_s, (self._request_timestamps[0] + 60.0) - now)
            if self.max_tokens_per_minute > 0:
                used = sum(tokens for _, tokens in self._recent_total_tokens)
                if used + expected_tokens > self.max_tokens_per_minute and self._recent_total_tokens:
                    wait_s = max(wait_s, (self._recent_total_tokens[0][0] + 60.0) - now)
            if wait_s > 0:
                time.sleep(min(wait_s, 5.0))
                continue
            break

    def _record_request_timestamp(self) -> None:
        now = time.monotonic()
        self._prune_windows(now)
        self._request_timestamps.append(now)

    def _record_total_tokens(self, total_tokens: int) -> None:
        now = time.monotonic()
        self._prune_windows(now)
        self._recent_total_tokens.append((now, max(1, int(total_tokens))))

    def _prune_windows(self, now: float) -> None:
        cutoff = now - 60.0
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()
        while self._recent_total_tokens and self._recent_total_tokens[0][0] < cutoff:
            self._recent_total_tokens.popleft()

    def _reduce_images_on_tpm_error(self, messages: list[dict[str, Any]], error_text: str) -> bool:
        if "tokens per min" not in error_text.lower():
            return False
        match = re.search(r"Limit\\s+(\\d+),\\s+Requested\\s+(\\d+)", error_text)
        if not match:
            return False
        limit = int(match.group(1))
        requested = int(match.group(2))
        if requested <= limit:
            return False
        keep_ratio = max(0.1, min(0.95, (limit / float(requested)) * 0.95))
        reduced_any = False
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            img_idx = [i for i, blk in enumerate(content) if isinstance(blk, dict) and blk.get("type") == "image_url"]
            n = len(img_idx)
            if n <= 1:
                continue
            keep = max(1, int(n * keep_ratio))
            if keep >= n:
                continue
            keep_ord = {
                int(round(i * (n - 1) / max(1, keep - 1))) if keep > 1 else 0
                for i in range(keep)
            }
            new_content: list[Any] = []
            seen = 0
            for i, blk in enumerate(content):
                if i in img_idx:
                    if seen in keep_ord:
                        new_content.append(blk)
                    seen += 1
                else:
                    new_content.append(blk)
            msg["content"] = new_content
            reduced_any = True
        return reduced_any


class GeminiNativeChatBackend:
    """Gemini backend using google-genai Files API + generate_content."""

    def __init__(self, config: dict[str, Any]):
        from google import genai

        self.config = config
        self.model_id = str(config["model_id"])
        self.temperature = float(config.get("temperature", 1.0))
        self.max_new_tokens = int(config.get("max_new_tokens", 192))
        self.api_key = config.get("api_key") or os.getenv(str(config.get("api_key_env_var", "GEMINI_API_KEY")))
        if not self.api_key:
            raise ValueError("Missing API key. Set GEMINI_API_KEY or pass api_key in backend config.")
        self.timeout_seconds = float(config.get("timeout_seconds", 180.0))
        self.min_seconds_between_requests = float(config.get("min_seconds_between_requests", 0.0))
        self.poll_interval_seconds = float(config.get("file_poll_interval_seconds", 2.0))
        self.debug_io = bool(config.get("debug_io", False))
        self.compact_history = bool(config.get("compact_history", True))
        self.thinking_budget = config.get("thinking_budget")
        self._last_request_at = 0.0
        self._client = genai.Client(api_key=self.api_key)
        self._uploaded_files_cache: dict[str, Any] = {}
        self._chat_sessions: dict[str, Any] = {}

    def generate(self, state: ConversationState, turn_index: int) -> BackendResponse:
        if self.min_seconds_between_requests > 0:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.min_seconds_between_requests - elapsed
            if remaining > 0:
                time.sleep(remaining)

        chat_key = _qwen_state_cache_key(state)
        chat = self._chat_sessions.get(chat_key)
        if chat is None:
            chat = self._create_chat_session(state)
            self._chat_sessions[chat_key] = chat

        response = chat.send_message(self._latest_user_message_parts(state))
        self._last_request_at = time.monotonic()
        if self.debug_io:
            self._debug_response("primary", response)

        usage = getattr(response, "usage_metadata", None)
        token_usage = None
        if usage is not None:
            prompt_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            completion_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
            total_tokens = int(getattr(usage, "total_token_count", 0) or 0)
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }

        text = (response.text or "").strip()
        if not _looks_like_complete_structured_answer(text):
            repaired = self._repair_response_format(
                chat=chat,
                draft_text=text,
            )
            if repaired:
                text = repaired
        return BackendResponse(text=text, token_usage=token_usage)

    def _repair_response_format(
        self,
        *,
        chat: Any,
        draft_text: str,
    ) -> str | None:
        if self.min_seconds_between_requests > 0:
            elapsed = time.monotonic() - self._last_request_at
            remaining = self.min_seconds_between_requests - elapsed
            if remaining > 0:
                time.sleep(remaining)

        repair_prompt = (
            "You omitted required fields. Re-output the answer exactly in this format only:\n"
            "ANSWER: <single letter>\n"
            "CONFIDENCE: <0-100>\n"
            "RATIONALE: <short paragraph>"
        )
        if draft_text:
            repair_prompt = f"Previous incomplete answer:\n{draft_text}\n\n{repair_prompt}"
        response = chat.send_message(
            repair_prompt,
            config=self._chat_config(temperature=0.0, max_output_tokens=max(self.max_new_tokens, 512)),
        )
        self._last_request_at = time.monotonic()
        if self.debug_io:
            self._debug_response("repair", response)
        fixed = (response.text or "").strip()
        if _looks_like_complete_structured_answer(fixed):
            return fixed
        return None

    def _debug_response(self, label: str, response: Any) -> None:
        try:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = None
            if candidates:
                finish_reason = getattr(candidates[0], "finish_reason", None)
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", None) if usage is not None else None
            completion_tokens = getattr(usage, "candidates_token_count", None) if usage is not None else None
            total_tokens = getattr(usage, "total_token_count", None) if usage is not None else None
            print(
                (
                    f"[Gemini debug:{label}] finish_reason={finish_reason} "
                    f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} total_tokens={total_tokens}"
                ),
                flush=True,
            )
            print(f"[Gemini debug:{label}] text={repr((response.text or '').strip())}", flush=True)
        except Exception as exc:  # pragma: no cover
            print(f"[Gemini debug:{label}] failed to print response debug: {exc}", flush=True)

    def _create_chat_session(self, state: ConversationState) -> Any:
        system_lines: list[str] = []
        for msg in state.messages:
            if str(msg.get("role", "user")) != "system":
                continue
            content = msg.get("content", [])
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    system_lines.append(str(block.get("text", "")))

        chat = self._client.chats.create(
            model=self.model_id,
            config=self._chat_config(system_instruction="\n".join(system_lines).strip() or None),
            history=[],
        )
        return chat

    def _latest_user_message_parts(self, state: ConversationState) -> list[Any] | str:
        from google.genai import types

        latest_user = next(
            (msg for msg in reversed(state.messages) if str(msg.get("role", "user")) == "user"),
            None,
        )
        if latest_user is None:
            return ""
        content = latest_user.get("content", [])
        if not isinstance(content, list):
            return str(content)
        return self._parts_from_blocks(content)

    def _parts_from_blocks(self, blocks: object) -> list[Any]:
        from google.genai import types

        if not isinstance(blocks, list):
            return [types.Part.from_text(text=str(blocks))]
        parts: list[Any] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", ""))
                if text:
                    parts.append(types.Part.from_text(text=text))
            elif block_type == "image":
                image_path = str(block.get("image", "")).strip()
                if image_path:
                    parts.append(
                        types.Part.from_bytes(
                            data=Path(image_path).expanduser().resolve().read_bytes(),
                            mime_type=_guess_image_mime_type(image_path),
                        )
                    )
            elif block_type == "video":
                video_path = str(block.get("video", "")).strip()
                if not video_path:
                    continue
                fps = float(block["fps"]) if block.get("fps") is not None else None
                max_frames = int(block["max_frames"]) if block.get("max_frames") is not None else None
                file_ref = self._get_or_upload_video_file(video_path)
                video_part = types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type)
                metadata_kwargs: dict[str, Any] = {}
                if fps is not None:
                    metadata_kwargs["fps"] = float(max(0.1, min(fps, 24.0)))
                if fps is not None and max_frames is not None and max_frames > 0:
                    metadata_kwargs["end_offset"] = f"{(max_frames / fps):.3f}s"
                if metadata_kwargs:
                    video_part.video_metadata = types.VideoMetadata(**metadata_kwargs)
                parts.append(video_part)
        return parts

    def _chat_config(
        self,
        *,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        system_instruction: str | None = None,
    ) -> Any:
        from google.genai import types

        output_tokens = max_output_tokens
        kwargs: dict[str, Any] = {
            "temperature": self.temperature if temperature is None else temperature,
            "max_output_tokens": output_tokens,
            "system_instruction": system_instruction,
        }
        if self.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=int(self.thinking_budget))
        return types.GenerateContentConfig(
            **kwargs,
        )

    def _build_gemini_contents(self, state: ConversationState) -> tuple[str, list[Any]]:
        from google.genai import types

        source_messages = state.messages
        if self.compact_history and len(state.messages) > 2:
            # Keep context compact for long multi-turn video chats:
            # - system prompt
            # - original user multimodal question (contains video)
            # - latest user follow-up instruction only
            first_user_idx = next(
                (idx for idx, msg in enumerate(state.messages) if str(msg.get("role", "user")) == "user"),
                None,
            )
            latest_user_idx = next(
                (idx for idx in range(len(state.messages) - 1, -1, -1) if str(state.messages[idx].get("role", "user")) == "user"),
                None,
            )
            compact: list[dict[str, object]] = []
            for msg in state.messages:
                if str(msg.get("role", "user")) == "system":
                    compact.append(msg)
                    break
            if first_user_idx is not None:
                compact.append(state.messages[first_user_idx])
            if latest_user_idx is not None and latest_user_idx != first_user_idx:
                latest_user = state.messages[latest_user_idx]
                content = latest_user.get("content", [])
                if isinstance(content, list):
                    text_only = [
                        block
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    compact.append({"role": "user", "content": text_only if text_only else content})
                else:
                    compact.append(latest_user)
            source_messages = compact

        system_instruction = ""
        contents: list[Any] = []

        for message in source_messages:
            role = str(message.get("role", "user"))
            raw_content = message.get("content", [])
            if not isinstance(raw_content, list):
                raw_content = [{"type": "text", "text": str(raw_content)}]

            if role == "system":
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        system_instruction += str(block.get("text", "")) + "\n"
                continue

            parts: list[Any] = []
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = str(block.get("text", ""))
                    if text:
                        parts.append(types.Part.from_text(text=text))
                elif block_type == "image":
                    image_path = str(block.get("image", "")).strip()
                    if image_path:
                        parts.append(
                            types.Part.from_bytes(
                                data=Path(image_path).expanduser().resolve().read_bytes(),
                                mime_type=_guess_image_mime_type(image_path),
                            )
                        )
                elif block_type == "video":
                    video_path = str(block.get("video", "")).strip()
                    if not video_path:
                        continue
                    fps = float(block["fps"]) if block.get("fps") is not None else None
                    max_frames = int(block["max_frames"]) if block.get("max_frames") is not None else None
                    file_ref = self._get_or_upload_video_file(video_path)
                    video_part = types.Part.from_uri(file_uri=file_ref.uri, mime_type=file_ref.mime_type)
                    metadata_kwargs: dict[str, Any] = {}
                    if fps is not None:
                        metadata_kwargs["fps"] = float(max(0.1, min(fps, 24.0)))
                    if fps is not None and max_frames is not None and max_frames > 0:
                        metadata_kwargs["end_offset"] = f"{(max_frames / fps):.3f}s"
                    if metadata_kwargs:
                        video_part.video_metadata = types.VideoMetadata(**metadata_kwargs)
                    parts.append(video_part)

            if not parts:
                continue
            mapped_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(role=mapped_role, parts=parts))

        return system_instruction.strip(), contents

    def _get_or_upload_video_file(self, video_path: str) -> Any:
        resolved = str(Path(video_path).expanduser().resolve())
        cached = self._uploaded_files_cache.get(resolved)
        if cached is not None:
            return cached

        file_ref = self._client.files.upload(file=resolved)
        while True:
            state = getattr(file_ref, "state", None)
            state_name = getattr(state, "name", state)
            state_text = str(state_name or "")
            if state_text == "ACTIVE":
                break
            if state_text == "FAILED":
                raise RuntimeError(f"Gemini file processing failed for {resolved}")
            time.sleep(self.poll_interval_seconds)
            file_ref = self._client.files.get(name=file_ref.name)

        self._uploaded_files_cache[resolved] = file_ref
        return file_ref


def _format_qwen_messages(messages: list[dict[str, object]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        converted_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                converted_content.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image":
                image_path = str(block.get("image", ""))
                if image_path:
                    converted_content.append({"type": "image", "image": image_path})
            elif block_type == "video":
                video_path = str(block.get("video", ""))
                if video_path:
                    video_block: dict[str, Any] = {"type": "video", "video": video_path}
                    if "fps" in block and block.get("fps") is not None:
                        video_block["fps"] = float(block["fps"])
                    if "max_frames" in block and block.get("max_frames") is not None:
                        video_block["max_frames"] = int(block["max_frames"])
                    if "nframes" in block and block.get("nframes") is not None:
                        video_block["nframes"] = int(block["nframes"])
                    converted_content.append(video_block)
        formatted.append(
            {
                "role": str(message.get("role", "user")),
                "content": converted_content,
            }
        )
    return formatted


def _format_internvl_messages(messages: list[dict[str, object]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", [])
        converted_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                converted_content.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image":
                image_path = str(block.get("image", ""))
                if image_path:
                    converted_content.append({"type": "image", "image": str(Path(image_path).resolve())})
        formatted.append(
            {
                "role": str(message.get("role", "user")),
                "content": converted_content,
            }
        )
    return formatted


def _collect_image_paths(messages: list[dict[str, Any]]) -> list[str]:
    images: list[str] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                image_path = str(block.get("image", ""))
                if image_path:
                    images.append(image_path)
    return images


def _collect_internvl_image_inputs(messages: list[dict[str, object]]) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                image_path = str(block.get("image", "")).strip()
                if image_path:
                    images.append(Image.open(image_path).convert("RGB"))
            elif block.get("type") == "video":
                video_path = str(block.get("video", "")).strip()
                if not video_path:
                    continue
                fps = block.get("fps")
                max_frames = block.get("max_frames")
                nframes = block.get("nframes")
                images.extend(
                    _decode_video_frames_for_internvl(
                        video_path=video_path,
                        fps=float(fps) if fps is not None else None,
                        max_frames=int(max_frames) if max_frames is not None else None,
                        nframes=int(nframes) if nframes is not None else None,
                    )
                )
    return images


def _prepare_internvl_messages_and_images(
    messages: list[dict[str, object]],
) -> tuple[list[dict[str, Any]], list[Any]]:
    from PIL import Image

    formatted: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        content = message.get("content", [])
        converted_content: list[dict[str, Any]] = []
        if not isinstance(content, list):
            formatted.append({"role": str(message.get("role", "user")), "content": converted_content})
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                converted_content.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image":
                image_path = str(block.get("image", "")).strip()
                if image_path:
                    images.append(Image.open(image_path).convert("RGB"))
                    # Placeholder used by InternVL chat template; processor consumes real image tensors.
                    converted_content.append({"type": "image", "image": "<image>"})
            elif block_type == "video":
                video_path = str(block.get("video", "")).strip()
                if not video_path:
                    continue
                decoded_frames = _decode_video_frames_for_internvl(
                    video_path=video_path,
                    fps=float(block["fps"]) if block.get("fps") is not None else None,
                    max_frames=int(block["max_frames"]) if block.get("max_frames") is not None else None,
                    nframes=int(block["nframes"]) if block.get("nframes") is not None else None,
                )
                if decoded_frames:
                    images.extend(decoded_frames)
                    converted_content.extend({"type": "image", "image": "<image>"} for _ in decoded_frames)
        formatted.append({"role": str(message.get("role", "user")), "content": converted_content})
    return formatted, images


def _decode_video_frames_for_internvl(
    *,
    video_path: str,
    fps: float | None,
    max_frames: int | None,
    nframes: int | None,
) -> list[Any]:
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    target_count = nframes if nframes and nframes > 0 else None
    if target_count is None and fps and fps > 0 and native_fps > 0:
        duration_sec = total_frames / native_fps
        target_count = max(1, int(duration_sec * fps))
    if target_count is None:
        target_count = 8
    if max_frames and max_frames > 0:
        target_count = min(target_count, max_frames)
    target_count = max(1, min(target_count, total_frames))

    if target_count == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / float(target_count - 1)
        indices = [int(round(i * step)) for i in range(target_count)]

    decoded: list[Any] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded.append(Image.fromarray(rgb))
    cap.release()
    return decoded


def _qwen_state_cache_key(state: ConversationState) -> str:
    return f"{state.example.question_id}:{state.strategy}"


def _resize_bgr_frame(frame: Any, max_side: int | None) -> Any:
    if max_side is None or max_side <= 0:
        return frame
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return frame
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_path_to_data_url(
    image_path: str,
    *,
    max_side: int | None = 512,
    jpeg_quality: int = 60,
    max_size_mb: float = 20.0,
    resize_factor: float = 0.75,
    min_side: int = 100,
) -> str:
    path = Path(image_path).expanduser().resolve()
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        # Fallback to raw-bytes path behavior if cv2 decode fails.
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{data}"
    frame = _resize_bgr_frame(frame, max_side)
    return _bgr_frame_to_jpeg_data_url(
        frame,
        max_side=None,
        jpeg_quality=jpeg_quality,
        max_size_mb=max_size_mb,
        resize_factor=resize_factor,
        min_side=min_side,
    )


def _guess_image_mime_type(image_path: str) -> str:
    ext = Path(image_path).suffix.lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/jpeg"


def _looks_like_complete_structured_answer(text: str) -> bool:
    upper = text.upper()
    if "ANSWER:" not in upper:
        return False
    if "CONFIDENCE:" not in upper:
        return False
    if "RATIONALE:" not in upper:
        return False
    # Require at least one digit after CONFIDENCE:
    confidence_tail = upper.split("CONFIDENCE:", 1)[1]
    if not any(ch.isdigit() for ch in confidence_tail[:12]):
        return False
    return True


def _bgr_frame_to_jpeg_data_url(
    frame: Any,
    *,
    max_side: int | None = 512,
    jpeg_quality: int = 60,
    max_size_mb: float = 20.0,
    resize_factor: float = 0.75,
    min_side: int = 100,
) -> str:
    frame = _resize_bgr_frame(frame, max_side)
    quality = int(max(20, min(95, jpeg_quality)))
    max_bytes = int(max(1.0, max_size_mb) * 1024 * 1024)
    scale_factor = float(max(0.1, min(0.99, resize_factor)))
    min_side_px = max(16, int(min_side))
    current = frame
    for _ in range(8):
        ok, encoded = cv2.imencode(".jpg", current, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode video frame as JPEG.")
        payload = encoded.tobytes()
        if len(payload) <= max_bytes:
            data = base64.b64encode(payload).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        h, w = current.shape[:2]
        if min(h, w) <= min_side_px:
            data = base64.b64encode(payload).decode("ascii")
            return f"data:image/jpeg;base64,{data}"
        new_w = max(min_side_px, int(round(w * scale_factor)))
        new_h = max(min_side_px, int(round(h * scale_factor)))
        current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)
    data = base64.b64encode(payload).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _video_to_frame_data_urls(
    *,
    video_path: str,
    fps: float | None,
    max_frames: int | None,
    nframes: int | None,
    image_max_side: int | None = 512,
    image_jpeg_quality: int = 60,
    image_max_size_mb: float = 20.0,
    image_resize_factor: float = 0.75,
    image_min_side: int = 100,
) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        cap.release()
        return []

    target_count = nframes if nframes and nframes > 0 else None
    if target_count is None and fps and fps > 0 and native_fps > 0:
        duration_sec = total_frames / native_fps
        target_count = max(1, int(duration_sec * fps))
    if target_count is None:
        target_count = 8
    if max_frames and max_frames > 0:
        target_count = min(target_count, max_frames)
    target_count = max(1, min(target_count, total_frames))

    if target_count == 1:
        indices = [0]
    else:
        step = (total_frames - 1) / float(target_count - 1)
        indices = [int(round(i * step)) for i in range(target_count)]

    data_urls: list[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        data_urls.append(
            _bgr_frame_to_jpeg_data_url(
                frame,
                max_side=image_max_side,
                jpeg_quality=image_jpeg_quality,
                max_size_mb=image_max_size_mb,
                resize_factor=image_resize_factor,
                min_side=image_min_side,
            )
        )
    cap.release()
    return data_urls


def _format_openai_compatible_messages(
    messages: list[dict[str, object]],
    *,
    image_detail: str = "low",
    image_max_side: int = 512,
    image_jpeg_quality: int = 60,
    image_max_size_mb: float = 20.0,
    image_resize_factor: float = 0.75,
    image_min_side: int = 100,
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", [])
        if not isinstance(content, list):
            formatted.append({"role": role, "content": str(content)})
            continue

        converted_content: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                converted_content.append({"type": "text", "text": str(block.get("text", ""))})
            elif block_type == "image":
                image_path = str(block.get("image", "")).strip()
                if image_path:
                    converted_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_path_to_data_url(
                                    image_path,
                                    max_side=image_max_side,
                                    jpeg_quality=image_jpeg_quality,
                                    max_size_mb=image_max_size_mb,
                                    resize_factor=image_resize_factor,
                                    min_side=image_min_side,
                                ),
                                "detail": image_detail,
                            },
                        }
                    )
            elif block_type == "video":
                video_path = str(block.get("video", "")).strip()
                if not video_path:
                    continue
                fps = block.get("fps")
                max_frames = block.get("max_frames")
                nframes = block.get("nframes")
                data_urls = _video_to_frame_data_urls(
                    video_path=video_path,
                    fps=float(fps) if fps is not None else None,
                    max_frames=int(max_frames) if max_frames is not None else None,
                    nframes=int(nframes) if nframes is not None else None,
                    image_max_side=image_max_side,
                    image_jpeg_quality=image_jpeg_quality,
                    image_max_size_mb=image_max_size_mb,
                    image_resize_factor=image_resize_factor,
                    image_min_side=image_min_side,
                )
                converted_content.extend(
                    {"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}}
                    for data_url in data_urls
                )
        formatted.append({"role": role, "content": converted_content})
    return formatted

