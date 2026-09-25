from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    pass


_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|analysis|reasoning)\b[^>]*>[\s\S]*?</(?:think|thinking|analysis|reasoning)\s*>",
    re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(
    r"</?(?:think|thinking|analysis|reasoning)\b[^>]*>",
    re.IGNORECASE,
)
_REASONING_UNCLOSED_RE = re.compile(
    r"<(?:think|thinking|analysis|reasoning)\b[^>]*>[\s\S]*$",
    re.IGNORECASE,
)


def clean_model_output(text: str) -> str:
    """Return only user-visible answer text, never tagged reasoning blocks."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_UNCLOSED_RE.sub("", text)
    text = _REASONING_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_retryable_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return name in {
        "apiconnectionerror",
        "apitimeouterror",
        "rate_limit_error",
        "internalservererror",
        "serviceunavailableerror",
        "timeout_error",
        "connectionerror",
    }


class AIClient:
    """Provider-agnostic OpenAI-compatible chat client with transient retries."""

    def __init__(self, api_key: str, base_url: str, timeout: float = 60.0) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content if response.choices else None
                content = clean_model_output(content or "")
                if not content:
                    raise AIError("AI returned an empty visible response")
                return content
            except AIError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < 2 and _is_retryable_error(exc):
                    delay = min(8.0, 1.5 * (attempt + 1))
                    logger.warning(
                        "AI request transient failure (%s); retrying in %.1fs (%d/3)",
                        type(exc).__name__, delay, attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.exception("AI request failed")
                break
        raise AIError("AI service request failed") from last_exc

    async def analyze_image(
        self,
        *,
        model: str,
        image_data_url: str,
        question: str,
        system_prompt: str,
    ) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question or "请分析这张图片。"},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ]
        return await self.chat(messages, model=model)
