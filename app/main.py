from __future__ import annotations

import asyncio
import logging
import random
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from telegram import (
    BotCommand,
    BotCommandScopeAllGroupChats,
    BotCommandScopeAllPrivateChats,
    Update,
)
from telegram.ext import Application

from .ai import AIClient
from .config import Settings, load_settings
from .database import Database
from .handlers import BotState, register_handlers, run_scheduled_group_summary
from .search import SearchService
from .utils import markdown_to_markdown_v2, split_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

IDLE_NUDGE_SONGS = (
    ("晴天", "周杰伦"),
    ("起风了", "买辣椒也用券"),
    ("有何不可", "许嵩"),
    ("我怀念的", "孙燕姿"),
    ("旅行的意义", "陈绮贞"),
    ("那些年", "胡夏"),
    ("稻香", "周杰伦"),
    ("平凡之路", "朴树"),
    ("想见你想见你想见你", "八三夭"),
    ("七里香", "周杰伦"),
)


def _normalize_language(language: str) -> str:
    value = (language or "en").strip().lower()
    if not re.fullmatch(r"[a-z-]{2,12}", value):
        return "en"
    return value


def _sort_event_year(event: dict[str, Any], target_year: int | None) -> tuple[int, int]:
    raw_year = event.get("year")
    try:
        event_year = int(raw_year)
    except (TypeError, ValueError):
        event_year = 0
    wanted = target_year or date.today().year
    return (0 if event_year else 1, abs(event_year - wanted) if event_year else 999999)


def _dedupe_and_filter_history(
    payload: dict[str, Any],
    year: int | None,
) -> list[dict[str, Any]]:
    buckets = (
        payload.get("selected", []),
        payload.get("events", []),
        payload.get("births", []),
        payload.get("deaths", []),
        payload.get("holidays", []),
    )
    events: list[dict[str, Any]] = []
    for bucket in buckets:
        if not isinstance(bucket, list):
            continue
        for item in bucket:
            if not isinstance(item, dict):
                continue
            raw_year = item.get("year")
            try:
                item_year = int(raw_year)
            except (TypeError, ValueError):
                item_year = None
            if year is not None and item_year != year:
                continue
            events.append(item)

    events.sort(key=lambda item: _sort_event_year(item, year))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for event in events:
        pages = event.get("pages") if isinstance(event.get("pages"), list) else []
        first_page = pages[0] if pages and isinstance(pages[0], dict) else {}
        titles = first_page.get("titles") if isinstance(first_page, dict) else {}
        canonical = titles.get("canonical") if isinstance(titles, dict) else first_page.get("title")
        key = (event.get("year"), event.get("text"), canonical)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
        if len(unique) >= 5:
            break
    return unique


async def _request_history_payload(
    client: httpx.AsyncClient,
    url: str,
) -> dict[str, Any] | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        logger.warning("History endpoint failed: %s (%s)", url, exc)
        return None


async def fetch_history_events(
    month: int,
    day: int,
    year: int | None,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Fetch Wikimedia On This Day data with two API paths and language fallback."""
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return []

    primary_language = _normalize_language(language)
    languages = [primary_language]
    if primary_language != "en":
        languages.append("en")

    headers = {
        "User-Agent": "telegram-ai-bot-v2/2.0 (+https://github.com/boxueduocai666/telegram-ai-bot-v2)",
        "Accept": "application/json",
    }
    timeout = httpx.Timeout(12.0, connect=8.0)
    async with httpx.AsyncClient(
        timeout=timeout,
        headers=headers,
        follow_redirects=True,
    ) as client:
        for lang in languages:
            endpoints = (
                f"https://{lang}.wikipedia.org/api/rest_v1/feed/onthisday/all/{month:02d}/{day:02d}",
                f"https://api.wikimedia.org/feed/v1/wikipedia/{lang}/onthisday/all/{month:02d}/{day:02d}",
            )
            for endpoint in endpoints:
                payload = await _request_history_payload(client, endpoint)
                if not payload:
                    continue
                events = _dedupe_and_filter_history(payload, year)
                if events:
                    logger.info(
                        "History data fetched successfully: language=%s endpoint=%s count=%s",
                        lang,
                        "central" if endpoint.startswith("https://api.wikimedia.org/") else "wikipedia",
                        len(events),
                    )
                    return events

    logger.error(
        "No usable Wikimedia history data for %02d/%02d (year=%s, language=%s)",
        month,
        day,
        year,
        primary_language,
    )
    return []


def render_history_events(
    month: int,
    day: int,
    year: int | None,
    events: list[dict[str, Any]],
) -> str:
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
        titles = page.get("titles") if isinstance(page, dict) else {}
        article = titles.get("normalized") if isinstance(titles, dict) else page.get("title")
        if not article and isinstance(page, dict):
            article = page.get("title")

        lines.append(f"{icons[index % len(icons)]} **{event_year}年**")
        if text:
            lines.append(text)
        if article:
            lines.append(f"来源：Wikipedia — {article}")
        lines.append("")

    lines.append("来源：Wikimedia / Wikipedia On This Day 数据")
    return "\n".join(lines).strip()


def _time_to_minutes(value: str) -> int | None:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", (value or "").strip())
    if not match:
        return None
    hour, minute = int(match.group(1)), int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _schedule_is_due(local_now: datetime, configured_time: str) -> bool:
    target = _time_to_minutes(configured_time)
    if target is None:
        logger.error("Invalid scheduled time in database: %r", configured_time)
        return False
    current = local_now.hour * 60 + local_now.minute
    return current >= target


def _parse_message_timestamp(raw: Any) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("UTC"))


def _known_group_settings(state: BotState) -> dict[int, Any]:
    merged: dict[int, Any] = {}
    for group in state.db.list_auto_summary_groups():
        merged[group.chat_id] = group
    for group in state.db.list_auto_history_groups():
        merged[group.chat_id] = group
    for chat_id in state.group_histories.keys():
        if chat_id not in merged:
            merged[chat_id] = state.db.get_group_settings(chat_id)
    return merged


async def _send_idle_nudge(state: BotState, chat_id: int, now_utc: datetime) -> bool:
    if state.application is None:
        return False

    rows = state.db.get_group_messages(chat_id, limit=1)
    if not rows:
        return False

    latest = rows[-1]
    try:
        latest_id = int(latest.get("message_id"))
    except (TypeError, ValueError):
        return False

    latest_activity = _parse_message_timestamp(latest.get("created_at"))
    if latest_activity is None:
        return False

    bot_activity = state.last_bot_activity.get(chat_id)
    if bot_activity and bot_activity > latest_activity:
        latest_activity = bot_activity

    if now_utc - latest_activity < timedelta(minutes=state.settings.idle_nudge_minutes):
        return False

    if getattr(state, "last_idle_nudge_message_id", {}).get(chat_id) == latest_id:
        return False

    last_song = getattr(state, "last_idle_song", {}).get(chat_id)
    candidates = [song for song in IDLE_NUDGE_SONGS if song != last_song]
    title, artist = random.choice(candidates or list(IDLE_NUDGE_SONGS))
    message = (
        f"检测到本群已冷场 {state.settings.idle_nudge_minutes} 分钟\n\n"
        "🎵 本机器人来营业一下：推荐一首好歌。\n"
        f"最近单曲循环的是《{title}》——{artist}。\n"
        "你们最近在听什么？"
    )

    try:
        await state.application.bot.send_message(chat_id=chat_id, text=message)
    except Exception:
        logger.exception("Idle group nudge failed for chat %s", chat_id)
        return False

    if not hasattr(state, "last_idle_nudge_message_id"):
        state.last_idle_nudge_message_id = {}
    if not hasattr(state, "last_idle_song"):
        state.last_idle_song = {}
    state.last_idle_nudge_message_id[chat_id] = latest_id
    state.last_idle_song[chat_id] = (title, artist)
    state.mark_bot_activity(chat_id, now_utc)
    logger.info("Idle nudge sent to chat %s", chat_id)
    return True


async def _send_scheduled_history(
    state: BotState, group: Any, date_key: str, local_now: datetime
) -> bool:
    events = await fetch_history_events(
        local_now.month,
        local_now.day,
        None,
        state.settings.history_language,
    )
    if not events or state.application is None:
        return False

    rendered = render_history_events(local_now.month, local_now.day, None, events)
    markdown_chunks = split_text(markdown_to_markdown_v2(rendered))
    plain_chunks = split_text(rendered)
    for index, chunk in enumerate(markdown_chunks):
        try:
            await state.application.bot.send_message(
                group.chat_id, chunk, parse_mode="MarkdownV2"
            )
        except Exception:
            logger.exception(
                "Scheduled history MarkdownV2 send failed for chat %s chunk %s",
                group.chat_id,
                index + 1,
            )
            fallback = plain_chunks[index] if index < len(plain_chunks) else chunk
            try:
                await state.application.bot.send_message(group.chat_id, fallback)
            except Exception:
                logger.exception(
                    "Scheduled history plain-text send failed for chat %s chunk %s",
                    group.chat_id,
                    index + 1,
                )
                return False

    if not state.db.mark_history_sent(group.chat_id, date_key):
        logger.error("Failed to mark scheduled history as sent for chat %s", group.chat_id)
        return False
    state.mark_bot_activity(group.chat_id)
    return True


async def background_scheduler(state: BotState, stop_event: asyncio.Event) -> None:
    logger.info("Background scheduler started")
    while not stop_event.is_set():
        try:
            now_utc = datetime.now(ZoneInfo("UTC"))
            known_groups = _known_group_settings(state)

            # 1) Optional automatic On This Day push.
            for group in list(state.db.list_auto_history_groups()):
                try:
                    local_now = now_utc.astimezone(ZoneInfo(group.timezone))
                    date_key = local_now.strftime("%Y-%m-%d")
                    if not _schedule_is_due(local_now, group.auto_history_time):
                        continue
                    if group.last_history_sent == date_key:
                        continue
                    await _send_scheduled_history(state, group, date_key, local_now)
                except (ZoneInfoNotFoundError, ValueError):
                    logger.exception("Invalid history timezone for chat %s", group.chat_id)
                except Exception:
                    logger.exception("Failed scheduled history for chat %s", group.chat_id)

            # 2) Daily 23:00 summary. This is independent of message count.
            for group in list(state.db.list_auto_summary_groups()):
                try:
                    local_now = now_utc.astimezone(ZoneInfo(group.timezone))
                    date_key = local_now.strftime("%Y-%m-%d")
                    if not _schedule_is_due(local_now, group.auto_summary_time):
                        continue
                    if group.last_summary_sent == date_key:
                        continue
                    if await run_scheduled_group_summary(state, group.chat_id, date_key):
                        logger.info("Daily summary sent for chat %s", group.chat_id)
                except (ZoneInfoNotFoundError, ValueError):
                    logger.exception("Invalid summary timezone for chat %s", group.chat_id)
                except Exception:
                    logger.exception("Failed scheduled summary for chat %s", group.chat_id)

            # 3) 15-minute idle nudge. It is independent of summary settings.
            for chat_id in set(known_groups) | set(state.group_histories.keys()):
                try:
                    await _send_idle_nudge(state, chat_id, now_utc)
                except Exception:
                    logger.exception("Failed idle nudge check for chat %s", chat_id)

        except asyncio.CancelledError:
            break
        except Exception:
            logger.exception("Background scheduler loop failed")

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=30)
        except asyncio.TimeoutError:
            pass

    logger.info("Background scheduler stopped")


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
    search = SearchService(
        settings.search_max_results,
        searxng_url=settings.searxng_url,
        region=settings.search_region,
        backend_timeout=settings.search_backend_timeout_seconds,
    )
    bot_state = BotState(settings, db, ai, search)

    bot_application = Application.builder().token(settings.telegram_bot_token).updater(None).build()
    register_handlers(bot_application, bot_state)

    await bot_application.initialize()
    await bot_application.start()
    me = await bot_application.bot.get_me()
    bot_state.bot_username = me.username or ""

    private_commands = [
        BotCommand("start", "开始使用"),
        BotCommand("help", "查看帮助"),
        BotCommand("model", "切换 AI 模型"),
        BotCommand("clear", "清除当前对话上下文"),
        BotCommand("search", "搜索互联网"),
        BotCommand("history", "查看历史上的今天"),
        BotCommand("status", "查看 Bot 状态"),
        BotCommand("ping", "测试 Bot 是否在线"),
        BotCommand("about", "关于这个 Bot"),
    ]
    group_commands = [
        BotCommand("help", "查看帮助"),
        BotCommand("model", "切换 AI 模型"),
        BotCommand("clear", "清除群聊 AI 上下文"),
        BotCommand("search", "搜索互联网"),
        BotCommand("summary", "总结当前群聊"),
        BotCommand("history", "查看历史上的今天"),
        BotCommand("status", "查看 Bot 状态"),
        BotCommand("ping", "测试 Bot 是否在线"),
        BotCommand("about", "关于这个 Bot"),
    ]
    try:
        await bot_application.bot.set_my_commands(
            private_commands,
            scope=BotCommandScopeAllPrivateChats(),
        )
        await bot_application.bot.set_my_commands(
            group_commands,
            scope=BotCommandScopeAllGroupChats(),
        )
    except Exception:
        logger.exception("Failed to register Telegram command menu; continuing startup")

    if not settings.public_url:
        logger.error("PUBLIC_URL/RENDER_EXTERNAL_URL/RAILWAY_PUBLIC_DOMAIN is not set; webhook registration skipped")
    else:
        await bot_application.bot.set_webhook(
            url=settings.webhook_url,
            secret_token=settings.webhook_secret,
            allowed_updates=Update.ALL_TYPES,
        )
        logger.info("Webhook configured at %s", settings.webhook_url)

    bot_state.application = bot_application
    # These dicts are intentionally in-memory. They only prevent duplicate idle
    # nudges within the current process and do not affect persisted summaries.
    bot_state.last_idle_nudge_message_id = {}
    bot_state.last_idle_song = {}
    stop_scheduler = asyncio.Event()
    scheduler_task = asyncio.create_task(background_scheduler(bot_state, stop_scheduler))

    try:
        yield
    finally:
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
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, bool]:
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

    uvicorn.run("app.main:app", host="0.0.0.0", port=load_settings().port)
