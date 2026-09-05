from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from telegram import Update
from telegram.ext import Application

from .ai import AIClient
from .config import Settings, load_settings
from .database import Database
from .handlers import BotState, register_handlers
from .utils import markdown_to_markdown_v2, split_text
from .search import SearchService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


async def fetch_history_events(month: int, day: int, year: int | None, language: str = "en") -> list[dict[str, Any]]:
    """Fetch Wikimedia On This Day data.

    Wikimedia documents /feed/onthisday/{type}/{mm}/{dd} as an experimental feed.
    Since the feed is undergoing a deprecation path, callers treat failure as a normal
    feature-level failure rather than crashing the bot.
    """
    endpoint = f"https://{language}.wikipedia.org/api/rest_v1/feed/onthisday/all/{month:02d}/{day:02d}"
    try:
        async with httpx.AsyncClient(timeout=12, headers={"User-Agent": "telegram-ai-bot-v2/2.0"}) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            data = response.json()
    except Exception:
        logger.exception("History API request failed")
        return []

    buckets = [data.get("selected", []), data.get("events", []), data.get("births", []), data.get("deaths", [])]
    events: list[dict[str, Any]] = []
    wanted_year = year
    for bucket in buckets:
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            item_year = item.get("year")
            if wanted_year is not None and item_year != wanted_year:
                continue
            events.append(item)
    # Prefer records with a year and a short human-readable text.
    events.sort(key=lambda item: (0 if item.get("year") is not None else 1, abs(int(item.get("year", 0)) - (year or date.today().year))))
    unique: list[dict[str, Any]] = []
    seen = set()
    for event in events:
        key = (event.get("year"), event.get("text"), event.get("pages", [{}])[0].get("title") if event.get("pages") else "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
        if len(unique) >= 5:
            break
    return unique


def render_history_events(month: int, day: int, year: int | None, events: list[dict[str, Any]]) -> str:
    title = f"📅 历史上的今天 · {month}月{day}日"
    if year:
        title += f" · {year}年"
    lines = [title, ""]
    icons = ["🏛️", "🚀", "💻", "🎨", "📚"]
    for index, event in enumerate(events):
        event_year = event.get("year", "")
        text = str(event.get("text", "")).strip()
        pages = event.get("pages") or []
        page = pages[0] if pages and isinstance(pages[0], dict) else {}
        article = page.get("titles", {}).get("canonical") if isinstance(page.get("titles"), dict) else page.get("title")
        lines.append(f"{icons[index % len(icons)]} **{event_year}年**")
        if text:
            lines.append(text)
        if article:
            lines.append(f"来源：Wikipedia — {article}")
        lines.append("")
    lines.append("来源：Wikimedia / Wikipedia On This Day 数据")
    return "\n".join(lines).strip()


async def history_scheduler(state: BotState, stop_event: asyncio.Event) -> None:
    logger.info("History scheduler started")
    while not stop_event.is_set():
        try:
            now_utc = datetime.now(ZoneInfo("UTC"))
            # Check at minute-level; all group calculations use their own timezone.
            for settings in state.db.list_auto_history_groups():
                try:
                    local_now = now_utc.astimezone(ZoneInfo(settings.timezone))
                    expected = settings.auto_history_time
                    current = local_now.strftime("%H:%M")
                    date_key = local_now.strftime("%Y-%m-%d")
                    if current != expected or settings.last_history_sent == date_key:
                        continue
                    events = await fetch_history_events(local_now.month, local_now.day, None, state.settings.history_language)
                    if not events:
                        continue
                    rendered = markdown_to_markdown_v2(render_history_events(local_now.month, local_now.day, None, events))
                    try:
                        for chunk in split_text(rendered):
                            await state.application.bot.send_message(settings.chat_id, chunk, parse_mode="MarkdownV2")  # type: ignore[union-attr]
                        await state.db.mark_history_sent(settings.chat_id, date_key)
                    except Exception:
                        logger.exception("Automatic history send failed for chat %s", settings.chat_id)
                        try:
                            for chunk in split_text(render_history_events(local_now.month, local_now.day, None, events)):
                                await state.application.bot.send_message(settings.chat_id, chunk)  # type: ignore[union-attr]
                            await state.db.mark_history_sent(settings.chat_id, date_key)
                        except Exception:
                            logger.exception("Automatic history plain-text fallback failed for chat %s", settings.chat_id)
                except Exception:
                    logger.exception("Failed scheduled history for chat %s", settings.chat_id)
        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("History scheduler loop failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass
    logger.info("History scheduler stopped")


# Runtime state is attached during lifespan. FastAPI's process is intentionally single-instance friendly.
settings: Settings | None = None
bot_application: Application | None = None
bot_state: BotState | None = None
scheduler_task: asyncio.Task[None] | None = None
stop_scheduler: asyncio.Event | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global settings, bot_application, bot_state, scheduler_task, stop_scheduler
    settings = load_settings()
    db = Database(settings.database_path, settings.default_timezone, settings.default_auto_time)
    db.initialize()
    ai = AIClient(settings.ai_api_key, settings.ai_base_url, settings.ai_timeout_seconds)
    search = SearchService(settings.search_max_results, searxng_url=settings.searxng_url)
    bot_state = BotState(settings, db, ai, search)

    bot_application = Application.builder().token(settings.telegram_bot_token).updater(None).build()
    register_handlers(bot_application, bot_state)

    # Explicitly initialize/start PTB with no update-fetching worker.
    await bot_application.initialize()
    await bot_application.start()
    me = await bot_application.bot.get_me()
    bot_state.bot_username = me.username or ""

    if not settings.public_url:
        logger.warning("PUBLIC_URL/RAILWAY_PUBLIC_DOMAIN is not set; webhook registration will be skipped")
    else:
        await bot_application.bot.set_webhook(
            url=settings.webhook_url,
            secret_token=settings.webhook_secret,
            allowed_updates=Update.ALL_TYPES,
        )
        logger.info("Webhook configured at %s", settings.webhook_url)

    # Used only for scheduler access; it is not a Telegram polling mechanism.
    bot_state.application = bot_application
    stop_scheduler = asyncio.Event()
    scheduler_task = asyncio.create_task(history_scheduler(bot_state, stop_scheduler))

    yield

    if stop_scheduler:
        stop_scheduler.set()
    if scheduler_task:
        await scheduler_task
    try:
        if settings and settings.public_url:
            await bot_application.bot.delete_webhook(drop_pending_updates=False)
    except Exception:
        logger.exception("Failed to delete webhook on shutdown")
    await bot_application.stop()
    await bot_application.shutdown()


app = FastAPI(title="Telegram AI Bot V2", version="2.0.0", lifespan=lifespan)


@app.get("/health", response_class=PlainTextResponse)
async def health() -> str:
    return "ok"


@app.post("/telegram/webhook")
async def telegram_webhook(request: Request, x_telegram_bot_api_secret_token: str | None = Header(default=None)) -> dict[str, bool]:
    if settings is None or bot_application is None:
        raise HTTPException(status_code=503, detail="Bot is not initialized")
    if x_telegram_bot_api_secret_token != settings.webhook_secret:
        raise HTTPException(status_code=403, detail="Forbidden")
    try:
        payload = await request.json()
        update = Update.de_json(payload, bot=bot_application.bot)
        if update is None:
            raise ValueError("Invalid Telegram update")
        await bot_application.update_queue.put(update)
        return {"ok": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Webhook update handling failed")
        raise HTTPException(status_code=400, detail="Invalid update") from exc


if __name__ == "__main__":
    import uvicorn

    # Railway provides PORT. Local default is 8080.
    uvicorn.run("app.main:app", host="0.0.0.0", port=load_settings().port)
