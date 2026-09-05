from __future__ import annotations

import logging
from typing import Any, Sequence

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AIError(RuntimeError):
    pass


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
        try:
            kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = await self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content if response.choices else None
            if not content:
                raise AIError("AI returned an empty response")
            return content.strip()
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
