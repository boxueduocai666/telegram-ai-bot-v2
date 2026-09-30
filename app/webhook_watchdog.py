from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

WATCH_INTERVAL_SECONDS = 30
STARTUP_RETRY_DELAYS = (0, 2, 5, 10, 20)
REPAIR_COOLDOWN_SECONDS = 60
REQUEST_TIMEOUT_SECONDS = 12

_watchdog_thread: threading.Thread | None = None
_watchdog_stop: threading.Event | None = None
_watchdog_lock = threading.Lock()


def _api_request(token: str, method: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict) or not data.get("ok"):
        raise RuntimeError(f"Telegram API {method} failed: {data!r}")
    result = data.get("result")
    return result if isinstance(result, dict) else {"value": result}


def _get_webhook_info(token: str) -> dict[str, Any]:
    return _api_request(token, "getWebhookInfo")


def _set_webhook(token: str, webhook_url: str, secret: str) -> None:
    _api_request(
        token,
        "setWebhook",
        {
            "url": webhook_url,
            "secret_token": secret,
            "allowed_updates": [
                "message",
                "edited_message",
                "callback_query",
                "channel_post",
                "edited_channel_post",
            ],
        },
    )


def _ensure_webhook(
    token: str,
    webhook_url: str,
    secret: str,
    *,
    force_repair_on_recent_error: bool = True,
) -> bool:
    """Verify the webhook and repair it when it is missing or unhealthy.

    Returns True only when this call actually performed a repair. A healthy
    webhook returns False so callers can use the return value for cooldowns.
    """
    info = _get_webhook_info(token)
    current_url = str(info.get("url") or "")
    pending = int(info.get("pending_update_count") or 0)
    last_error = str(info.get("last_error_message") or "")

    missing_or_wrong = current_url != webhook_url
    recent_delivery_problem = bool(last_error and pending > 0) and force_repair_on_recent_error
    if not missing_or_wrong and not recent_delivery_problem:
        return False

    if missing_or_wrong:
        reason = f"webhook mismatch: current={current_url!r}, expected={webhook_url!r}"
    else:
        reason = f"recent delivery error with pending updates: {last_error!r}"
    logger.warning("Webhook needs repair: %s", reason)

    _set_webhook(token, webhook_url, secret)
    verified = _get_webhook_info(token)
    verified_url = str(verified.get("url") or "")
    if verified_url != webhook_url:
        raise RuntimeError(
            f"Webhook repair verification failed: current={verified_url!r}, expected={webhook_url!r}"
        )

    logger.info(
        "Webhook self-healed successfully: url=%s pending_update_count=%s",
        webhook_url,
        verified.get("pending_update_count", 0),
    )
    return True


def _watch_loop(
    token: str,
    webhook_url: str,
    secret: str,
    stop_event: threading.Event,
) -> None:
    logger.info("Webhook watchdog started: interval=%ss url=%s", WATCH_INTERVAL_SECONDS, webhook_url)
    last_repair_at = 0.0
    first_check = True

    while not stop_event.is_set():
        try:
            now = time.monotonic()
            force_repair = (now - last_repair_at) >= REPAIR_COOLDOWN_SECONDS
            repaired = _ensure_webhook(
                token,
                webhook_url,
                secret,
                force_repair_on_recent_error=force_repair,
            )
            if first_check:
                logger.info("Webhook watchdog initial verification completed")
                first_check = False
            if repaired:
                last_repair_at = time.monotonic()
        except Exception:
            logger.exception("Webhook watchdog check failed; will retry automatically")

        stop_event.wait(WATCH_INTERVAL_SECONDS)

    logger.info("Webhook watchdog stopped")


def start_webhook_watchdog(token: str, webhook_url: str, secret: str) -> None:
    """Start exactly one daemon watchdog for the current Python process."""
    global _watchdog_thread, _watchdog_stop

    if not token or not webhook_url.startswith(("http://", "https://")):
        logger.warning("Webhook watchdog not started: invalid token or webhook URL")
        return

    with _watchdog_lock:
        if _watchdog_thread is not None and _watchdog_thread.is_alive():
            return

        _watchdog_stop = threading.Event()
        _watchdog_thread = threading.Thread(
            target=_watch_loop,
            args=(token, webhook_url, secret, _watchdog_stop),
            name="telegram-webhook-watchdog",
            daemon=True,
        )
        _watchdog_thread.start()
        logger.info("Telegram Webhook auto-repair enabled")


def stop_webhook_watchdog() -> None:
    """Stop the current process' watchdog cleanly during application shutdown."""
    global _watchdog_thread, _watchdog_stop

    with _watchdog_lock:
        stop_event = _watchdog_stop
        thread = _watchdog_thread
        _watchdog_stop = None
        _watchdog_thread = None

    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=2.0)
