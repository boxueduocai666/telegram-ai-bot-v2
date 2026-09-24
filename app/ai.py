from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    pass


# Some reasoning-capable providers put their hidden reasoning into the normal
# message content. Never forward these tagged blocks to Telegram users.
_REASONING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|analysis|reasoning)\b[^>]*>[\s\S]*?</(?:think|thinking|analysis|reasoning)\s*>",
    re.IGNORECASE,
)
_REASONING_TAG_RE = re.compile(
    r"</?(?:think|thinking|analysis|reasoning)\b[^>]*>",
    re.IGNORECASE,
)


def clean_model_output(text: str) -> str:
    """Return only user-visible answer text, never tagged reasoning blocks."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_retryable_error(exc: Exception) -> bool:
    """Identify transient provider failures that are safe to retry once."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and (
        status == 408 or status == 409 or status == 429 or status >= 500
    ):
        return True
    name = exc.__class__.__name__.lower()
    return any(
        marker in name
        for marker in (
            "timeout",
            "connection",
            "network",
            "temporarilyunavailable",
        )
    )


class AIClient:
    """Provider-agnostic OpenAI-compatible chat client."""

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

        try:
            response = None
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    response = await self.client.chat.completions.create(**kwargs)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == 1 or not _is_retryable_error(exc):
                        raise
                    logger.warning(
                        "Transient AI request failure; retrying once: %s", exc
                    )
                    await asyncio.sleep(1.0)

            if last_error is not None or response is None:
                raise last_error or AIError("AI request returned no response")

            content = response.choices[0].message.content if response.choices else None
            content = clean_model_output(content or "")
            if not content:
                raise AIError("AI returned an empty visible response")
            return content
        except AIError:
            raise
        except Exception as exc:
            logger.exception("AI request failed")
            raise AIError("AI service request failed") from exc

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
