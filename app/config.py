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


def _parse_models(raw: str, default_model: str) -> tuple[str, ...]:
    """Parse model names robustly, including full-width commas and semicolons.

    This prevents a malformed Render environment such as using Chinese commas
    from becoming one giant Telegram button label.
    """
    tokens = re.split(r"[,，;；\n\r]+", raw or "")
    models: list[str] = []
    for token in tokens:
        model = token.strip().strip('"\'[]')
        if not model:
            continue
        # Also tolerate accidental JSON-like quoting around individual names.
        model = model.strip()
        if model and model not in models:
            models.append(model)
    if default_model and default_model not in models:
        models.insert(0, default_model)
    return tuple(models)


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
    history_language: str = "en"
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
    model = _env("DEFAULT_MODEL")
    secret = _env("WEBHOOK_SECRET")

    # DEFAULT_MODEL is optional on Render. When it is omitted, fall back to
    # AVAILABLE_MODELS (first entry), then to the V2 default model. This keeps
    # the service bootable while still allowing an explicit override.
    models_raw = _env("AVAILABLE_MODELS", "agnes-2.0-flash")
    parsed_models = _parse_models(models_raw, "")
    if not model:
        model = parsed_models[0] if parsed_models else "agnes-2.0-flash"

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

    models_raw = _env("AVAILABLE_MODELS", model)
    models = _parse_models(models_raw, model)

    public_url = _env(
        "PUBLIC_URL",
        _env("RENDER_EXTERNAL_URL", _env("RAILWAY_PUBLIC_DOMAIN")),
    )

    return Settings(
        telegram_bot_token=token,
        webhook_secret=secret,
        ai_api_key=api_key,
        ai_base_url=base_url,
        default_model=model,
        available_models=models,
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
        history_language=_env("HISTORY_LANGUAGE", "en"),
        searxng_url=_env("SEARXNG_URL") or None,
        search_region=_env("SEARCH_REGION", "wt-wt"),
        search_backend_timeout_seconds=max(3, _int_env("SEARCH_BACKEND_TIMEOUT_SECONDS", 8)),
        idle_nudge_minutes=max(1, _int_env("IDLE_NUDGE_MINUTES", 15)),
    )
