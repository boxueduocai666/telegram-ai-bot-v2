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
_REASONING_TAG_RE = re.compile(r"</?(?:think|thinking|analysis|reasoning)\b[^>]*>", re.IGNORECASE)
_REASONING_UNCLOSED_RE = re.compile(
    r"<(?:think|thinking|analysis|reasoning)\b[^>]*>[\s\S]*$", re.IGNORECASE
)


def clean_model_output(text: str) -> str:
    """Remove common tagged reasoning blocks before sending text to Telegram."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _REASONING_BLOCK_RE.sub("", text)
    text = _REASONING_UNCLOSED_RE.sub("", text)
    text = _REASONING_TAG_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_gemini_model(model: str) -> bool:
    value = (model or "").strip().lower()
    return value.startswith("gemini-") or value.startswith("models/gemini-")


def _is_retryable_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 425, 429, 500, 502, 503, 504}:
        return True
    return name in {
        "apiconnectionerror",
        "apitimeouterror",
        "ratelimiterror",
        "rate_limit_error",
        "internalservererror",
        "serviceunavailableerror",
        "timeout_error",
        "connectionerror",
    }


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


class AIClient:
    """OpenAI-compatible client with separate Gemini routing and safe text failover."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        timeout: float = 60.0,
        *,
        gemini_api_key: str = "",
        gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
        gemini_models: Sequence[str] = (),
        agnes_models: Sequence[str] = (),
        fallback_models: Sequence[str] = (),
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.gemini_models = _unique(tuple(gemini_models))
        self.agnes_models = _unique(tuple(agnes_models))
        self.fallback_models = _unique(
            tuple(fallback_models) + self.agnes_models + self.gemini_models
        )
        self.gemini_client = (
            AsyncOpenAI(api_key=gemini_api_key.strip(), base_url=gemini_base_url, timeout=timeout)
            if gemini_api_key.strip()
            else None
        )
        self._catalog_cache: dict[str, tuple[str, ...]] = {}

    def _client_for_model(self, model: str) -> AsyncOpenAI:
        if _is_gemini_model(model):
            if self.gemini_client is None:
                raise AIError("Gemini API is not configured (GEMINI_API_KEY is missing)")
            return self.gemini_client
        return self.client

    async def list_models(self, provider: str, *, refresh: bool = False) -> tuple[str, ...]:
        """Return configured model IDs plus supported live catalogue entries.

        Gateways that do not implement GET /models still return their configured
        model list, which keeps the Telegram model picker usable.
        """
        normalized_provider = "gemini" if provider.strip().lower() == "gemini" else "agnes"
        configured = self.gemini_models if normalized_provider == "gemini" else self.agnes_models
        client = self.gemini_client if normalized_provider == "gemini" else self.client

        if normalized_provider == "gemini" and client is None:
            return configured
        if not refresh and normalized_provider in self._catalog_cache:
            return _unique(tuple(configured) + self._catalog_cache[normalized_provider])

        discovered: list[str] = []
        try:
            response = await asyncio.wait_for(client.models.list(), timeout=8.0)
            for item in getattr(response, "data", ()) or ():
                model_id = str(getattr(item, "id", "") or "").strip()
                if not model_id:
                    continue
                if normalized_provider == "gemini" and _is_gemini_model(model_id):
                    discovered.append(model_id)
                elif normalized_provider == "agnes" and not _is_gemini_model(model_id):
                    discovered.append(model_id)
        except Exception as exc:
            # A compatible gateway may support chat completions but not GET /models.
            logger.info(
                "Could not discover %s models; using configured model list (%s)",
                normalized_provider,
                type(exc).__name__,
            )

        combined = _unique(tuple(configured) + tuple(sorted(discovered)))
        self._catalog_cache[normalized_provider] = combined
        return combined

    @staticmethod
    def _text_only(messages: Sequence[dict[str, Any]]) -> bool:
        return not any(isinstance(message.get("content"), list) for message in messages)

    async def _request_model(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        *,
        temperature: float | None,
        max_tokens: int | None,
        attempts: int,
    ) -> str:
        client = self._client_for_model(model)
        kwargs: dict[str, Any] = {"model": model, "messages": list(messages)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        last_exc: Exception | None = None
        for attempt in range(max(1, attempts)):
            try:
                response = await client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content if response.choices else None
                content = clean_model_output(content or "")
                if not content:
                    raise AIError(f"Model {model} returned an empty response")
                return content
            except AIError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt + 1 < attempts and _is_retryable_error(exc):
                    delay = min(8.0, 1.5 * (attempt + 1))
                    logger.warning(
                        "AI request failed for model %s (%s); retrying in %.1fs (%d/%d)",
                        model, type(exc).__name__, delay, attempt + 1, attempts,
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.warning("AI request failed for model %s (%s)", model, type(exc).__name__)
                break
        raise AIError(f"AI request failed for model {model}") from last_exc

    async def chat(
        self,
        messages: Sequence[dict[str, Any]],
        model: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        try:
            return await self._request_model(
                messages, model, temperature=temperature, max_tokens=max_tokens, attempts=3
            )
        except AIError as primary_error:
            # Never send image payloads to a different provider/model automatically.
            if not self._text_only(messages):
                raise

            primary_provider = "gemini" if _is_gemini_model(model) else "agnes"
            candidates = list(self.fallback_models)
            for provider in (primary_provider, "agnes" if primary_provider == "gemini" else "gemini"):
                if provider == "gemini" and self.gemini_client is None:
                    continue
                try:
                    candidates.extend(await self.list_models(provider, refresh=True))
                except Exception:
                    logger.debug("Model catalogue refresh failed for %s", provider, exc_info=True)

            candidates = list(_unique(tuple(candidates)))
            same_provider = [
                candidate for candidate in candidates
                if candidate != model
                and ("gemini" if _is_gemini_model(candidate) else "agnes") == primary_provider
            ]
            other_provider = [candidate for candidate in candidates if candidate != model and candidate not in same_provider]

            for candidate in (same_provider + other_provider)[:3]:
                try:
                    answer = await self._request_model(
                        messages,
                        candidate,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        attempts=1,
                    )
                    logger.warning("Automatic model failover: %s -> %s", model, candidate)
                    return f"🛟 主模型 {model} 暂不可用，本次已自动切换备用模型 {candidate}。\n\n{answer}"
                except AIError:
                    logger.info("Fallback model %s also failed", candidate)

            raise primary_error

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
