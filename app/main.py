from __future__ import annotations

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from telegram import BotCommand, BotCommandScopeAllGroupChats, BotCommandScopeAllPrivateChats, Update
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


# Runtime state is attached during lifespan. FastAPI's process is intentionally single-instance friendly.
settings: Settings | None = None
bot_application: Application | None = None
bot_state: BotState | None = None
scheduler_task: asyncio.Task[None] | None = None
stop_scheduler: asyncio.Event | None = None

# In-memory state for the anti-spam part of the idle nudge feature. A restart
# intentionally resets these values; the bot will still wait a full idle period
# before sending again based on the newest persisted group message.
_idle_nudge_last_message_id: dict[int, int] = {}
_last_bot_activity: dict[int, datetime] = {}
_last_idle_song: dict[int, tuple[str, str]] = {}


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
    """Collect groups known by the process/database without adding DB schema.

    Existing V2 groups are discoverable through the scheduled-history/summary
    settings. Fresh groups are also held in BotState.group_histories as soon as
    a meaningful message is seen.
    """
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
    rows = state.db.get_group_messages(chat_id, limit=1)
    if not rows:
        return False

    latest = rows[-1]
    latest_id_raw = latest.get("message_id")
    try:
        latest_id = int(latest_id_raw)
    except (TypeError, ValueError):
        return False

    latest_activity = _parse_message_timestamp(latest.get("created_at"))
    if latest_activity is None:
        return False

    bot_activity = _last_bot_activity.get(chat_id)
    if bot_activity and bot_activity > latest_activity:
        latest_activity = bot_activity

    if now_utc - latest_activity < timedelta(minutes=state.settings.idle_nudge_minutes):
        return False

    # One nudge per latest user message. A new meaningful message automatically
    # changes latest_id and starts a fresh 15-minute window.
    if _idle_nudge_last_message_id.get(chat_id) == latest_id:
        return False

    candidates = [song for song in IDLE_NUDGE_SONGS if song != _last_idle_song.get(chat_id)]
    song = random.choice(candidates or list(IDLE_NUDGE_SONGS))
    title, artist = song
    message = (
        f"检测到本群已冷场 {state.settings.idle_nudge_minutes} 分钟\n\n"
        f"🎵 本机器人来营业一下：推荐一首好歌。\n"
        f"最近单曲循环的是《{title}》——{artist}。\n"
        "你们最近在听什么？"
    )

    try:
        await state.application.bot.send_message(chat_id=chat_id, text=message)  # type: ignore[union-attr]
    except Exception:
        logger.exception("Idle group nudge failed for chat %s", chat_id)
        return False

    _idle_nudge_last_message_id[chat_id] = latest_id
    _last_idle_song[chat_id] = song
    _last_bot_activity[chat_id] = now_utc
    logger.info("Idle nudge sent to chat %s", chat_id)
    return True


async def background_scheduler(state: BotState, stop_event: asyncio.Event) -> None:
    logger.info("Background scheduler started")
    while not stop_event.is_set():
        try:
            now_utc = datetime.now(ZoneInfo("UTC"))
            group_settings = _known_group_settings(state)

            # 1) Existing automatic 'On This Day' pushes.
            for group in list(state.db.list_auto_history_groups()):
                try:
                    local_now = now_utc.astimezone(ZoneInfo(group.timezone))
                    expected = group.auto_history_time
                    current = local_now.strftime("%H:%M")
                    date_key = local_now.strftime("%Y-%m-%d")
                    if current != expected or group.last_history_sent == date_key:
                        continue
                    events = await fetch_history_events(local_now.month, local_now.day, None, state.settings.history_language)
                    if not events:
                        continue
                    rendered = markdown_to_markdown_v2(render_history_events(local_now.month, local_now.day, None, events))
                    sent = False
                    try:
                        for chunk in split_text(rendered):
                            await state.application.bot.send_message(group.chat_id, chunk, parse_mode="MarkdownV2")  # type: ignore[union-attr]
                        sent = True
                    except Exception:
                        logger.exception("Automatic history send failed for chat %s", group.chat_id)
                        try:
                            for chunk in split_text(render_history_events(local_now.month, local_now.day, None, events)):
                                await state.application.bot.send_message(group.chat_id, chunk)  # type: ignore[union-attr]
                            sent = True
                        except Exception:
                            logger.exception("Automatic history plain-text fallback failed for chat %s", group.chat_id)
                    if sent and state.db.mark_history_sent(group.chat_id, date_key):
                        _last_bot_activity[group.chat_id] = now_utc
                except Exception:
                    logger.exception("Failed scheduled history for chat %s", group.chat_id)

            # 2) Existing V2 daily group summary scheduler: independent of the
            # old 30-message trigger. It runs at each group's configured local time
            # (default 23:00 Asia/Shanghai).
            for group in list(state.db.list_auto_summary_groups()):
                try:
                    local_now = now_utc.astimezone(ZoneInfo(group.timezone))
                    expected = group.auto_summary_time
                    current = local_now.strftime("%H:%M")
                    date_key = local_now.strftime("%Y-%m-%d")
                    if current != expected or group.last_summary_sent == date_key:
                        continue
                    if await run_scheduled_group_summary(state, group.chat_id, date_key):
                        _last_bot_activity[group.chat_id] = now_utc
                except Exception:
                    logger.exception("Failed scheduled summary for chat %s", group.chat_id)

            # 3) New idle-room nudge. It runs independently of /summary auto on/off.
            # A group must have at least one meaningful persisted message first.
            for chat_id in set(group_settings) | set(state.group_histories.keys()):
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
    # Register Telegram's native command menu so users can type "/" and pick
    # commands instead of having to memorize them. Keep private/group menus
    # slightly different to avoid cluttering group chats with /start.
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
    await bot_application.bot.set_my_commands(
        private_commands, scope=BotCommandScopeAllPrivateChats()
    )
    await bot_application.bot.set_my_commands(
        group_commands, scope=BotCommandScopeAllGroupChats()
    )
    logger.info("Telegram command menu registered")
    if not settings.public_url:
        logger.warning("PUBLIC_URL/RENDER_EXTERNAL_URL is not set; webhook registration will be skipped")
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
    scheduler_task = asyncio.create_task(background_scheduler(bot_state, stop_scheduler))

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

    # Render and Railway both provide PORT. Local default is 8080.
    uvicorn.run("app.main:app", host="0.0.0.0", port=load_settings().port)
