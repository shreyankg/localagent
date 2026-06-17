"""MLX model wrapper — model-agnostic LLM engine."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

logger = logging.getLogger(__name__)


class Engine:
    """Lazy-loading wrapper around an MLX language model.

    The model is loaded on first use, not at construction time.
    Supports any MLX-compatible model via ``model_path``.
    """

    def __init__(
        self,
        model_path: str = "mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> None:
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("Loading model: %s", self.model_path)
            self._model, self._tokenizer = load(self.model_path)
            logger.info("Model loaded successfully")

    # -- prompt formatting ---------------------------------------------------

    @staticmethod
    def _fold_system_into_user(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Merge any leading system message into the first user message.

        Some models (e.g. Gemma 2) don't support a ``system`` role in their
        chat template.  This helper prepends the system content to the first
        user turn so the instructions are still seen by the model.
        """
        if not messages or messages[0]["role"] != "system":
            return messages

        system_content = messages[0]["content"]
        rest = messages[1:]
        merged: list[dict[str, str]] = []
        injected = False
        for msg in rest:
            if not injected and msg["role"] == "user":
                merged.append({
                    "role": "user",
                    "content": f"{system_content}\n\n{msg['content']}",
                })
                injected = True
            else:
                merged.append(msg)
        # Edge case: system message but no user message
        if not injected:
            merged.insert(0, {"role": "user", "content": system_content})
        return merged

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        """Format messages using the tokenizer's chat template.

        ``messages`` is a list of ``{"role": ..., "content": ...}`` dicts.
        If the template rejects the system role, the system message is
        automatically folded into the first user message and retried.
        """
        self._ensure_loaded()
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as exc:
                if "system" in str(exc).lower():
                    logger.debug(
                        "System role not supported by template, folding into user message"
                    )
                    folded = self._fold_system_into_user(messages)
                    return self._tokenizer.apply_chat_template(
                        folded, tokenize=False, add_generation_prompt=True
                    )
                raise
        # Fallback for models without a chat template
        parts: list[str] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    # -- generation ----------------------------------------------------------

    def generate_text(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text from a list of chat messages.

        Returns the raw model output as a string.
        """
        self._ensure_loaded()
        prompt = self._apply_chat_template(messages)
        temp = temperature if temperature is not None else self.temperature
        sampler = make_sampler(temp=temp)
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or self.max_tokens,
            sampler=sampler,
            verbose=False,
        )
        return response.strip()

    def generate_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Generate and parse a JSON response from the model.

        Extracts JSON from the model output (handles markdown code fences),
        validates it parses correctly, and retries on failure.

        Retry strategy:
        - 1st retry: append a correction hint so the model can fix near-misses.
        - Subsequent retries: fresh attempt with original messages to avoid
          bloating context with prior failed output.

        Raises ``ValueError`` if JSON cannot be extracted after retries.
        """
        last_error: Exception | None = None
        total_attempts = 1 + retries
        original_messages = messages

        for attempt in range(total_attempts):
            raw = self.generate_text(messages, max_tokens=max_tokens)

            try:
                return self._extract_json(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "JSON parse failed (attempt %d/%d): %s",
                    attempt + 1,
                    total_attempts,
                    exc,
                )
                if attempt == 0 and retries >= 1:
                    # First retry: correction hint with failed output
                    messages = [
                        *original_messages,
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Please respond with ONLY valid JSON, no other text."
                            ),
                        },
                    ]
                else:
                    # Subsequent retries: fresh attempt
                    messages = original_messages

        raise ValueError(
            f"Failed to get valid JSON after {total_attempts} attempts: {last_error}"
        ) from last_error

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Extract JSON from model output, handling code fences."""
        # Try to find JSON in a code block first
        fence_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL
        )
        if fence_match:
            return json.loads(fence_match.group(1))

        # Try to find a raw JSON object
        brace_start = text.find("{")
        if brace_start != -1:
            # Find the matching closing brace
            depth = 0
            for i, ch in enumerate(text[brace_start:], start=brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[brace_start : i + 1])

        raise ValueError("No JSON object found in model output")
