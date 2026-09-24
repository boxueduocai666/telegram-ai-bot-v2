from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

WATCH_INTERVAL_SECONDS = 30
STARTUP_RETRY_DELAYS = (0, 2, 5, 10, 20)
REPAIR_COOLDOWN_SECONDS = 60

_watchdog_thread: threading.Thread | None = None
_watchdog_lock = threading.Lock()


def _telegram_api_url(token: str, method: str) -> str:
    return f"https://api.telegram.org/bot{token}/{method}"


def _api_call(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    body = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        _telegram_api_url(token, method),
        data=body,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("ok"):
        description = data.get("description", "unknown Telegram API error") if isinstance(data, dict) else "invalid response"
        raise RuntimeError(f"Telegram {method} failed: {description}")
    return data


def _get_webhook_info(token: str) -> dict[str, Any]:
    data = _api_call(token, "getWebhookInfo")
    result = data.get("result")
    return result if isinstance(result, dict) else {}


def _allowed_updates() -> list[str]:
    # Keep exactly the same update coverage as the main application.
    try:
        from telegram import Update

        return list(Update.ALL_TYPES)
    except Exception:
        # Safe fallback for an early-start situation where PTB cannot be imported yet.
        return ["message", "edited_message", "channel_post", "edited_channel_post", "callback_query"]


def _set_webhook(token: str, webhook_url: str, secret: str) -> None:
    _api_call(
        token,
        "setWebhook",
        {
            "url": webhook_url,
            "secret_token": secret,
            "allowed_updates": _allowed_updates(),
        },
    )


def _ensure_webhook(token: str, webhook_url: str, secret: str, *, force_repair_on_recent_error: bool = True) -> bool:
    info = _get_webhook_info(token)
    current_url = str(info.get("url") or "")
    pending = int(info.get("pending_update_count") or 0)
    last_error = str(info.get("last_error_message") or "")

    missing_or_wrong = current_url != webhook_url
    recent_delivery_problem = bool(last_error and pending > 0) and force_repair_on_recent_error

    if not missing_or_wrong and not recent_delivery_problem:
        return True

    reason = "missing/mismatched URL" if missing_or_wrong else "pending updates with a recent delivery error"
    logger.warning("Webhook self-heal triggered (%s)", reason)
    _set_webhook(token, webhook_url, secret)

    verified = _get_webhook_info(token)
    verified_url = str(verified.get("url") or "")
    if verified_url != webhook_url:
        raise RuntimeError(
            f"Webhook repair returned success but verification failed: expected {webhook_url}, got {verified_url or '<empty>'}"
        )

    logger.info(
        "Webhook self-healed successfully: url=%s pending_update_count=%s",
        webhook_url,
        verified.get("pending_update_count", 0),
    )
    return True


def _watch_loop(token: str, webhook_url: str, secret: str) -> None:
    logger.info("Webhook watchdog started: interval=%ss url=%s", WATCH_INTERVAL_SECONDS, webhook_url)

    last_repair_at = 0.0
    first_check = True

    while True:
        try:
            now = time.monotonic()
            force_repair = (now - last_repair_at) >= REPAIR_COOLDOWN_SECONDS
            _ensure_webhook(
                token,
                webhook_url,
                secret,
                force_repair_on_recent_error=force_repair,
            )
            if first_check:
                logger.info("Webhook watchdog initial verification passed")
                first_check = False
            last_repair_at = now if force_repair and first_check is False else last_repair_at
        except Exception:
            logger.exception("Webhook watchdog check failed; will retry automatically")

        time.sleep(WATCH_INTERVAL_SECONDS)


def start_webhook_watchdog(token: str, webhook_url: str, secret: str) -> None:
    """Start exactly one daemon watchdog for the current Python process."""
    global _watchdog_thread

    if not token or not webhook_url.startswith(("http://", "https://")):
        return

    with _watchdog_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return

        _watchdog_thread = threading.Thread(
            target=_watch_loop,
            args=(token, webhook_url, secret),
            name="telegram-webhook-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()
        logger.info("Telegram Webhook auto-repair enabled")
