import ast
from pathlib import Path

from app.webhook_watchdog import _ensure_webhook, start_webhook_watchdog, stop_webhook_watchdog


def test_project_python_syntax():
    project_root = Path(__file__).resolve().parents[1]
    for source_file in (project_root / "app").glob("*.py"):
        ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))


def test_watchdog_healthy_webhook_does_not_repair(monkeypatch):
    calls = []

    def fake_info(_token):
        return {
            "url": "https://example.com/telegram/webhook",
            "pending_update_count": 0,
            "last_error_message": "",
        }

    monkeypatch.setattr("app.webhook_watchdog._get_webhook_info", fake_info)
    monkeypatch.setattr("app.webhook_watchdog._set_webhook", lambda *args: calls.append(args))

    assert _ensure_webhook("token", "https://example.com/telegram/webhook", "secret") is False
    assert calls == []


def test_watchdog_repairs_wrong_webhook(monkeypatch):
    responses = [
        {"url": "https://old.example.com/webhook", "pending_update_count": 0},
        {"url": "https://example.com/telegram/webhook", "pending_update_count": 0},
    ]
    calls = []

    def fake_info(_token):
        return responses.pop(0)

    monkeypatch.setattr("app.webhook_watchdog._get_webhook_info", fake_info)
    monkeypatch.setattr("app.webhook_watchdog._set_webhook", lambda *args: calls.append(args))

    assert _ensure_webhook("token", "https://example.com/telegram/webhook", "secret") is True
    assert calls == [("token", "https://example.com/telegram/webhook", "secret")]


def test_watchdog_stop_is_idempotent(monkeypatch):
    monkeypatch.setattr(
        "app.webhook_watchdog._ensure_webhook",
        lambda *args, **kwargs: False,
    )
    start_webhook_watchdog("token", "https://example.com/telegram/webhook", "secret")
    stop_webhook_watchdog()
    stop_webhook_watchdog()
