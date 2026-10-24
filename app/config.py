from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _parse_models(raw: str) -> tuple[str, ...]:
    """Parse model IDs separated by commas, semicolons, or newlines."""
    tokens = re.split(r"[,，;；\n\r]+", raw or "")
    result: list[str] = []
    for token in tokens:
        model = token.strip().strip('"\'[]').strip()
        if model and model not in result:
            result.append(model)
    return tuple(result)


def _is_gemini_model(model: str) -> bool:
    value = model.strip().lower()
    return value.startswith("gemini-") or value.startswith("models/gemini-")


def _unique(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value.strip() for value in values if value.strip()))


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    webhook_secret: str
    ai_api_key: str
    ai_base_url: str
    default_model: str
    available_models: tuple[str, ...] = field(default_factory=tuple)
    database_path: str = "data/bot.db"
    webhook_path: str = "/telegram/webhook"
    public_url: str = ""
    port: int = 8080
    default_timezone: str = "Asia/Shanghai"
    default_auto_time: str = "23:00"
    context_max_messages: int = 20
    context_max_chars: int = 30000
    max_reply_context_length: int = 8000
    max_input_chars: int = 12000
    max_output_chars: int = 30000
    search_max_results: int = 5
    ai_timeout_seconds: float = 60.0
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    gemini_models: tuple[str, ...] = field(default_factory=tuple)
    history_language: str = "zh"
    searxng_url: str | None = None
    search_region: str = "wt-wt"
    search_backend_timeout_seconds: int = 8
    idle_nudge_minutes: int = 15

    @property
    def webhook_url(self) -> str:
        base = self.public_url.rstrip("/")
        if base and not base.startswith(("http://", "https://")):
            base = "https://" + base
        return f"{base}{self.webhook_path}" if base else self.webhook_path


def load_settings() -> Settings:
    token = _env("TELEGRAM_BOT_TOKEN")
    api_key = _env("AI_API_KEY")
    base_url = _env("AI_BASE_URL").rstrip("/")
    secret = _env("WEBHOOK_SECRET")

    gemini_api_key = _env("GEMINI_API_KEY")
    gemini_base_url = (
        _env("GEMINI_BASE_URL")
        or "https://generativelanguage.googleapis.com/v1beta/openai/"
    ).rstrip("/") + "/"

    # Preserve the previous V2 default while accepting both old and new settings.
    legacy_models = _parse_models(_env("AVAILABLE_MODELS") or "agnes-2.0-flash")
    agnes_extra = _parse_models(_env("AGNES_MODELS"))
    configured_models = _unique(list(legacy_models) + list(agnes_extra))

    # These are real Gemini API model IDs. Extra IDs in GEMINI_MODELS are additive.
    default_gemini_models = (
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.8-flash",
    )
    configured_gemini = _parse_models(_env("GEMINI_MODELS"))
    gemini_models = _unique(
        list(default_gemini_models)
        + [model for model in configured_models if _is_gemini_model(model)]
        + list(configured_gemini)
    )
    agnes_models = _unique(
        [model for model in configured_models if not _is_gemini_model(model)]
    )

    model = _env("DEFAULT_MODEL")
    if not model:
        model = next(iter(agnes_models), next(iter(gemini_models), "agnes-2.0-flash"))
    if _is_gemini_model(model):
        gemini_models = _unique(list(gemini_models) + [model])
    else:
        agnes_models = _unique(list(agnes_models) + [model])
    if not agnes_models:
        agnes_models = ("agnes-2.0-flash",)

    missing = [
        name
        for name, value in (
            ("TELEGRAM_BOT_TOKEN", token),
            ("AI_API_KEY", api_key),
            ("AI_BASE_URL", base_url),
            ("WEBHOOK_SECRET", secret),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    public_url = _env(
        "PUBLIC_URL",
        _env("RENDER_EXTERNAL_URL", _env("RENDER_EXTERNAL_HOSTNAME", _env("RAILWAY_PUBLIC_DOMAIN"))),
    )
    return Settings(
        telegram_bot_token=token,
        webhook_secret=secret,
        ai_api_key=api_key,
        ai_base_url=base_url,
        default_model=model,
        available_models=agnes_models,
        database_path=_env("DATABASE_PATH", "data/bot.db"),
        webhook_path=_env("WEBHOOK_PATH", "/telegram/webhook"),
        public_url=public_url,
        port=_int_env("PORT", 8080),
        default_timezone=_env("DEFAULT_TIMEZONE", "Asia/Shanghai"),
        default_auto_time=_env("DEFAULT_AUTO_TIME", "23:00"),
        context_max_messages=max(1, _int_env("CONTEXT_MAX_MESSAGES", 20)),
        context_max_chars=max(1000, _int_env("CONTEXT_MAX_CHARS", 30000)),
        max_reply_context_length=max(500, _int_env("MAX_REPLY_CONTEXT_LENGTH", 8000)),
        max_input_chars=max(1000, _int_env("MAX_INPUT_CHARS", 12000)),
        max_output_chars=max(1000, _int_env("MAX_OUTPUT_CHARS", 30000)),
        search_max_results=max(1, _int_env("SEARCH_MAX_RESULTS", 5)),
        ai_timeout_seconds=max(5.0, _float_env("AI_TIMEOUT_SECONDS", 60.0)),
        gemini_api_key=gemini_api_key,
        gemini_base_url=gemini_base_url,
        gemini_models=gemini_models,
        history_language=_env("HISTORY_LANGUAGE", "zh"),
        searxng_url=_env("SEARXNG_URL") or None,
        search_region=_env("SEARCH_REGION", "wt-wt"),
        search_backend_timeout_seconds=max(3, _int_env("SEARCH_BACKEND_TIMEOUT_SECONDS", 8)),
        idle_nudge_minutes=max(1, _int_env("IDLE_NUDGE_MINUTES", 15)),
    )
