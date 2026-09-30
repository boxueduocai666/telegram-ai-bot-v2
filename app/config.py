from __future__ import annotations

import os
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


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    webhook_secret: str
    ai_api_key: str
    ai_base_url: str
    default_model: str
    available_models: tuple[str, ...] = field(default_factory=tuple)
    database_path: str = "/data/bot.db"
    webhook_path: str = "/telegram/webhook"
    public_url: str = ""
    port: int = 8080
    default_timezone: str = "Asia/Shanghai"
    default_auto_time: str = "08:00"
    context_max_messages: int = 20
    context_max_chars: int = 30000
    max_reply_context_length: int = 8000
    max_input_chars: int = 12000
    max_output_chars: int = 30000
    search_max_results: int = 5
    ai_timeout_seconds: float = 60.0
    history_language: str = "en"
    searxng_url: str | None = None

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
    model = _env("DEFAULT_MODEL")
    secret = _env("WEBHOOK_SECRET")

    missing = [
        name
        for name, value in (
            ("TELEGRAM_BOT_TOKEN", token),
            ("AI_API_KEY", api_key),
            ("AI_BASE_URL", base_url),
            ("DEFAULT_MODEL", model),
            ("WEBHOOK_SECRET", secret),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    models_raw = _env("AVAILABLE_MODELS", model)
    models = tuple(dict.fromkeys(x.strip() for x in models_raw.split(",") if x.strip()))
    if model not in models:
        models = (model, *models)

    return Settings(
        telegram_bot_token=token,
        webhook_secret=secret,
        ai_api_key=api_key,
        ai_base_url=base_url,
        default_model=model,
        available_models=models,
        database_path=_env("DATABASE_PATH", "/data/bot.db"),
        webhook_path=_env("WEBHOOK_PATH", "/telegram/webhook"),
        public_url=_env("PUBLIC_URL", _env("RAILWAY_PUBLIC_DOMAIN")),
        port=_int_env("PORT", 8080),
        default_timezone=_env("DEFAULT_TIMEZONE", "Asia/Shanghai"),
        default_auto_time=_env("DEFAULT_AUTO_TIME", "08:00"),
        context_max_messages=_int_env("CONTEXT_MAX_MESSAGES", 20),
        context_max_chars=_int_env("CONTEXT_MAX_CHARS", 30000),
        max_reply_context_length=_int_env("MAX_REPLY_CONTEXT_LENGTH", 8000),
        max_input_chars=_int_env("MAX_INPUT_CHARS", 12000),
        max_output_chars=_int_env("MAX_OUTPUT_CHARS", 30000),
        search_max_results=_int_env("SEARCH_MAX_RESULTS", 5),
        ai_timeout_seconds=float(_env("AI_TIMEOUT_SECONDS", "60")),
        history_language=_env("HISTORY_LANGUAGE", "en"),
        searxng_url=_env("SEARXNG_URL") or None,
    )
