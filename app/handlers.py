from __future__ import annotations

import asyncio
import asyncio
import logging
import re
import time
from collections import defaultdict, deque
from datetime import date, datetime
from zoneinfo import ZoneInfo
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ChatType
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .ai import AIClient, AIError
from .config import Settings
from .database import Database, GroupSettings
from .image_generation import ImageGenerationError, ImageGenerationService, parse_image_request
from .image_search import ImageSearchError, ImageSearchService
from .search import SearchService, format_search_results
from .summary import ChatLine, summarize_chat
from .utils import (
    ReplyContext,
    build_reply_context,
    markdown_to_markdown_v2,
    normalize_text,
    parse_time,
    truncate_text,
    valid_timezone,
    split_text,
)
from .vision import VisionError, analyze_telegram_image

logger = logging.getLogger(__name__)
MAX_STORED_CONTEXT_MESSAGES = 40
MAX_GROUP_SUMMARY_MESSAGES = 120
MIN_SUMMARY_MESSAGES = 3
IGNORED_GROUP_MESSAGES = {
    "你好", "嗨", "哈喽", "hello", "hi", "早", "早上好", "晚上好", "晚安",
    "哈哈", "哈哈哈", "哈哈哈哈", "嗯", "哦", "噢", "啊", "好的", "好", "收到",
    "ok", "OK", "666", "6",
}


class BotState:
    def __init__(self, settings: Settings, db: Database, ai: AIClient, search: SearchService) -> None:
        self.settings = settings
        self.db = db
        self.ai = ai
        self.search = search
        self.bot_username = ""
        self.application: Application | None = None
        self.contexts: dict[str, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_STORED_CONTEXT_MESSAGES)
        )
        self.group_contexts: dict[int, deque[dict[str, str]]] = defaultdict(
            lambda: deque(maxlen=MAX_STORED_CONTEXT_MESSAGES)
        )
        self.group_histories: dict[int, deque[ChatLine]] = defaultdict(
            lambda: deque(maxlen=MAX_GROUP_SUMMARY_MESSAGES)
        )
        self.summary_in_progress: set[int] = set()
        self.last_bot_activity: dict[int, datetime] = {}
        self.chat_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self.last_idle_nudge_message_id: dict[int, int] = {}
        self.last_idle_song: dict[int, tuple[str, str]] = {}

    def lock_for(self, update: Update) -> asyncio.Lock:
        return self.chat_locks[self.context_key(update)]

    def context_key(self, update: Update) -> str:
        if not update.effective_chat or not update.effective_user:
            return "unknown"
        return (
            f"{update.effective_chat.id}:{update.effective_user.id}"
            if update.effective_chat.type == ChatType.PRIVATE
            else str(update.effective_chat.id)
        )

    def clear_context(self, update: Update) -> None:
        if update.effective_chat and update.effective_chat.type != ChatType.PRIVATE:
            self.group_contexts.pop(update.effective_chat.id, None)
        else:
            self.contexts.pop(self.context_key(update), None)

    def recent_context(self, update: Update) -> list[dict[str, str]]:
        if update.effective_chat and update.effective_chat.type != ChatType.PRIVATE:
            bucket = self.group_contexts[update.effective_chat.id]
        else:
            bucket = self.contexts[self.context_key(update)]
        return list(bucket)[-self.settings.context_max_messages:]

    def add_context(self, update: Update, user_text: str, assistant_text: str) -> None:
        if update.effective_chat and update.effective_chat.type != ChatType.PRIVATE:
            bucket = self.group_contexts[update.effective_chat.id]
        else:
            bucket = self.contexts[self.context_key(update)]
        bucket.append({"role": "user", "content": truncate_text(user_text, 6000)})
        bucket.append({"role": "assistant", "content": truncate_text(assistant_text, 10000)})

    def add_group_message(
        self,
        chat_id: int,
        author: str,
        text: str,
        message_id: int | None = None,
        chat_username: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        clean = truncate_text(normalize_text(text), 4000)
        if not clean:
            return
        self.group_contexts[chat_id].append({
            "role": "user",
            "content": f"{author or '用户'}：{clean}",
        })
        self.group_histories[chat_id].append(
            ChatLine(
                role="user",
                text=clean,
                author=author or "用户",
                message_id=message_id,
                chat_id=chat_id,
                chat_username=chat_username,
                timestamp=timestamp,
            )
        )

    def add_group_assistant(self, chat_id: int, text: str) -> None:
        clean = truncate_text(normalize_text(text), 10000)
        if clean:
            self.group_contexts[chat_id].append({"role": "assistant", "content": clean})

    def clear_group_history_through(self, chat_id: int, through_message_id: int) -> None:
        bucket = self.group_histories.get(chat_id)
        if not bucket:
            return
        kept = [line for line in bucket if not line.message_id or line.message_id > through_message_id]
        bucket.clear()
        bucket.extend(kept)


WELCOME = """🤖 *Telegram AI Bot V2*

一个面向长期维护的 Telegram AI 助手，支持多轮对话、图片理解、联网搜索、群聊总结与历史上的今天。"""

HELP_TEXT = """🤖 *Telegram AI Bot V2*

私聊：直接发送消息即可。
群聊：使用 @机器人或回复机器人消息后提问。
命令：
/start  开始使用
/help   查看帮助
/model  切换 AI 模型
/clear  清除当前上下文
/search <关键词>  强制联网搜索
/imagesearch <内容>  搜索互联网图片
/image <描述>  使用 AI 生成图片
/image 2.1 <描述>  使用 Image 2.1 生成图片
/summary  总结当前群聊
/summary auto  查看每日自动总结状态
/summary auto on|off  管理员开关每日自动总结
/summary auto 23:00  管理员设置每日总结时间
/history  历史上的今天
/history 8月8日
/history 2008-08-08
/history auto  查看自动推送状态
/history auto on|off  管理员开关自动推送
/history auto 08:00  管理员设置时间
/history timezone Asia/Shanghai  管理员设置时区
/status  查看服务状态
/ping  测试响应
/about  关于 Bot
"""



async def _telegram_call_with_retry(operation: Any, *args: Any, attempts: int = 3, **kwargs: Any) -> Any:
    """Retry transient Telegram API failures without masking permanent errors."""
    attempts = max(1, attempts)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation(*args, **kwargs)
        except RetryAfter as exc:
            last_error = exc
            delay = max(1.0, float(getattr(exc, "retry_after", 1.0)))
            if attempt + 1 >= attempts:
                raise
            logger.warning("Telegram rate limited; retrying in %.1fs (%d/%d)", delay, attempt + 1, attempts)
            await asyncio.sleep(delay)
        except (TimedOut, NetworkError) as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            delay = min(8.0, 1.5 * (attempt + 1))
            logger.warning("Telegram transient error %s; retrying in %.1fs (%d/%d)", type(exc).__name__, delay, attempt + 1, attempts)
            await asyncio.sleep(delay)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Telegram operation failed")


async def telegram_error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last-resort isolation so one broken update never crashes the bot process."""
    logger.exception("Unhandled Telegram update exception", exc_info=context.error)
    message = update.effective_message if update else None
    if not message:
        return
    try:
        await _telegram_call_with_retry(
            message.reply_text,
            "⚠️ 这条消息处理失败了，Bot 仍在运行，请稍后重试。",
        )
    except Exception:
        logger.exception("Failed to send global Telegram error notice")


def register_handlers(application: Application, state: BotState) -> None:
    handlers: list[Any] = [
        CommandHandler("start", start_command),
        CommandHandler("help", help_command),
        CommandHandler("model", model_command),
        CommandHandler("clear", clear_command),
        CommandHandler("search", search_command),
        CommandHandler("imagesearch", image_search_command),
        CommandHandler("image", image_command),
        CommandHandler("summary", summary_command),
        CommandHandler("history", history_command),
        CommandHandler("status", status_command),
        CommandHandler("ping", ping_command),
        CommandHandler("about", about_command),
        CallbackQueryHandler(model_callback, pattern=r"^model:"),
        MessageHandler(filters.PHOTO | filters.TEXT & ~filters.COMMAND, message_handler),
    ]
    for handler in handlers:
        application.add_handler(handler)
    application.bot_data["state"] = state
    application.add_error_handler(telegram_error_handler)


def get_state(context: ContextTypes.DEFAULT_TYPE) -> BotState:
    return context.application.bot_data["state"]


async def send_markdown(update: Update, text: str) -> None:
    if not update.effective_message:
        return
    rendered_chunks = split_text(markdown_to_markdown_v2(text))
    plain_chunks = split_text(text)
    for index, chunk in enumerate(rendered_chunks, start=1):
        try:
            if index == 1:
                await _telegram_call_with_retry(
                    update.effective_message.reply_text,
                    chunk,
                    parse_mode="MarkdownV2",
                )
            else:
                await _telegram_call_with_retry(
                    update.effective_message.chat.send_message,
                    chunk,
                    parse_mode="MarkdownV2",
                )
        except Exception:
            logger.exception("MarkdownV2 send failed for chunk %s; falling back to plain text", index)
            fallback = plain_chunks[index - 1] if index - 1 < len(plain_chunks) else chunk
            try:
                if index == 1:
                    await _telegram_call_with_retry(update.effective_message.reply_text, fallback)
                else:
                    await _telegram_call_with_retry(update.effective_message.chat.send_message, fallback)
            except Exception:
                logger.exception("Plain-text send also failed for chunk %s", index)


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send_markdown(update, WELCOME)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("🤖 AI模型", callback_data="model:menu")],
            [InlineKeyboardButton("🔎 联网搜索", callback_data="help:search")],
            [InlineKeyboardButton("📅 历史上的今天", callback_data="help:history")],
            [InlineKeyboardButton("📝 总结群聊", callback_data="help:summary")],
            [InlineKeyboardButton("🧹 清除上下文", callback_data="help:clear")],
            [InlineKeyboardButton("ℹ️ 关于", callback_data="help:about")],
        ]
    )
    if update.effective_message:
        try:
            await update.effective_message.reply_text(
                markdown_to_markdown_v2(HELP_TEXT), parse_mode="MarkdownV2", reply_markup=keyboard
            )
        except Exception:
            await update.effective_message.reply_text(HELP_TEXT, reply_markup=keyboard)


async def model_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_model_menu(update, get_state(context))


async def show_model_menu(update: Update, state: BotState) -> None:
    current = state.db.get_user_model(update.effective_user.id, state.settings.default_model) if update.effective_user else state.settings.default_model
    rows = []
    for model in state.settings.available_models:
        label = f"✅ {model}" if model == current else model
        rows.append([InlineKeyboardButton(label, callback_data=f"model:set:{model}")])
    markup = InlineKeyboardMarkup(rows)
    if update.callback_query:
        await update.callback_query.edit_message_text(f"当前模型：{current}\n\n请选择模型：", reply_markup=markup)
    elif update.effective_message:
        await update.effective_message.reply_text(f"当前模型：{current}\n\n请选择模型：", reply_markup=markup)


async def model_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    state = get_state(context)
    if not query or not update.effective_user:
        return
    await query.answer()
    data = query.data or ""
    if data == "model:menu":
        await show_model_menu(update, state)
        return
    _, _, model = data.partition("model:set:")
    if model not in state.settings.available_models:
        await query.edit_message_text("这个模型当前不可用。")
        return
    if state.db.set_user_model(update.effective_user.id, model):
        await query.edit_message_text(f"✅ 已切换模型：{model}")
    else:
        await query.edit_message_text("⚠️ 模型保存失败，请稍后重试。")


async def clear_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(context)
    state.clear_context(update)
    await update.effective_message.reply_text("🧹 当前对话上下文已清除。")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(context)
    message = update.effective_message
    if not message:
        return
    query = normalize_text(" ".join(context.args))
    if not query:
        await _telegram_call_with_retry(message.reply_text, "用法：/search 关键词")
        return

    status = await _telegram_call_with_retry(message.reply_text, "🔎 正在搜索并整理结果……")
    try:
        await _telegram_call_with_retry(message.chat.send_action, ChatAction.TYPING)
    except Exception:
        logger.warning("Unable to send search typing action", exc_info=True)

    try:
        results = await asyncio.wait_for(state.search.search(query), timeout=55.0)
        if not results:
            await _telegram_call_with_retry(status.edit_text, "🔎 当前无法获取联网搜索结果，请稍后再试。")
            return
        current = state.db.get_user_model(
            update.effective_user.id if update.effective_user else 0,
            state.settings.default_model,
        )
        prompt = (
            "你正在进行一次强制联网搜索回答。只能把搜索结果作为外部资料使用；不要声称访问了不存在的页面。\n"
            "不要输出 <think>、<thinking>、<analysis>、<reasoning> 或任何内部推理过程，只输出最终给用户看的答案。\n\n"
            f"用户问题：{query}\n\n搜索结果：\n{format_search_results(results)}"
        )
        answer = await asyncio.wait_for(
            state.ai.chat(
                [
                    {"role": "system", "content": "你是严谨的联网搜索问答助手。只输出最终答案。"},
                    {"role": "user", "content": prompt},
                ],
                model=current,
                max_tokens=3500,
            ),
            timeout=65.0,
        )
        chunks = split_text(markdown_to_markdown_v2(answer))
        plain_chunks = split_text(answer)
        if not chunks:
            await _telegram_call_with_retry(status.edit_text, "（AI 返回了空消息）")
            return
        try:
            await _telegram_call_with_retry(status.edit_text, chunks[0], parse_mode="MarkdownV2")
        except Exception:
            logger.exception("Search answer MarkdownV2 edit failed")
            fallback = plain_chunks[0] if plain_chunks else answer
            await _telegram_call_with_retry(status.edit_text, fallback)
            chunks = chunks[1:]
        else:
            chunks = chunks[1:]
        for index, chunk in enumerate(chunks, start=1):
            try:
                await _telegram_call_with_retry(
                    message.chat.send_message,
                    chunk,
                    parse_mode="MarkdownV2",
                )
            except Exception:
                fallback_index = index if index < len(plain_chunks) else len(plain_chunks) - 1
                fallback = plain_chunks[fallback_index] if plain_chunks and fallback_index >= 0 else chunk
                await _telegram_call_with_retry(message.chat.send_message, fallback)
    except asyncio.TimeoutError:
        logger.warning("Search timed out for query %r", query)
        await _telegram_call_with_retry(status.edit_text, "⚠️ 搜索或整理超时，请稍后再试。")
    except AIError:
        logger.exception("Search AI answer failed")
        await _telegram_call_with_retry(status.edit_text, "⚠️ AI 服务暂时不可用，搜索结果本身未伪造成成功回答。")
    except Exception:
        logger.exception("Search command failed")
        await _telegram_call_with_retry(status.edit_text, "⚠️ 联网搜索暂时出现问题，请稍后重试。")


async def image_search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return
    query = normalize_text(" ".join(context.args))
    if not query:
        await message.reply_text("用法：/imagesearch 关键词")
        return

    status = await message.reply_text("🔎 正在搜索图片……")
    try:
        service = ImageSearchService(max_results=5)
        results = await service.search(query)
        if not results:
            await status.edit_text("⚠️ 没有找到可用的图片结果，请换个关键词再试。")
            return

        lines = [f"🔎 图片搜索：{query}", ""]
        buttons: list[list[InlineKeyboardButton]] = []
        for index, result in enumerate(results, start=1):
            lines.append(f"{index}. {result.title}")
            buttons.append([InlineKeyboardButton("🖼️ 点击查看图片", url=result.image_url)])

        await status.edit_text(
            "\n".join(lines),
            reply_markup=InlineKeyboardMarkup(buttons),
        )
    except ImageSearchError:
        logger.exception("Image search failed for query %r", query)
        await status.edit_text("⚠️ 图片搜索暂时不可用，请稍后再试。")
    except Exception:
        logger.exception("Unexpected image search handler failure")
        await status.edit_text("⚠️ 图片搜索失败，请稍后再试。")


async def image_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if not message:
        return

    try:
        model, prompt = parse_image_request(context.args)
    except ImageGenerationError as exc:
        await message.reply_text(f"用法：/image <描述>\n可选模型：/image 2.1 <描述>\n\n⚠️ {exc}")
        return

    status = await message.reply_text("🎨 正在生成图片……")
    try:
        state = get_state(context)
        service = ImageGenerationService(
            state.settings.ai_api_key,
            state.settings.ai_base_url,
            timeout=120.0,
        )
        image_url = await service.generate(prompt, model)
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🖼️ 点击查看图片", url=image_url)]]
        )
        await status.edit_text("🎨 图片生成完成", reply_markup=keyboard)
    except ImageGenerationError:
        logger.exception("Image generation failed for model %s", model)
        await status.edit_text("⚠️ 图片生成失败，请稍后再试。")
    except Exception:
        logger.exception("Unexpected image generation handler failure")
        await status.edit_text("⚠️ 图片生成失败，请稍后再试。")


async def _load_summary_lines(state: BotState, chat_id: int) -> list[ChatLine]:
    rows = state.db.get_group_messages(chat_id, limit=120)
    if rows:
        lines: list[ChatLine] = []
        for row in rows:
            timestamp = None
            raw_timestamp = row.get("created_at")
            if raw_timestamp:
                try:
                    timestamp = datetime.fromisoformat(raw_timestamp)
                except ValueError:
                    timestamp = None
            lines.append(
                ChatLine(
                    role="user",
                    text=str(row.get("text") or ""),
                    author=str(row.get("author") or "用户"),
                    message_id=int(row["message_id"]) if row.get("message_id") is not None else None,
                    chat_id=int(row["chat_id"]) if row.get("chat_id") is not None else chat_id,
                    chat_username=row.get("chat_username"),
                    timestamp=timestamp,
                )
            )
        return lines
    return list(state.group_histories.get(chat_id, []))


def _is_meaningful_group_message(text: str) -> bool:
    cleaned = normalize_text(text)
    if not cleaned or cleaned in IGNORED_GROUP_MESSAGES or len(cleaned) <= 1:
        return False
    if re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
        return False
    return True


async def _send_markdown_to_chat(chat: Any, text: str) -> None:
    rendered_chunks = split_text(markdown_to_markdown_v2(text))
    plain_chunks = split_text(text)
    for index, chunk in enumerate(rendered_chunks):
        try:
            await _telegram_call_with_retry(chat.send_message, chunk, parse_mode="MarkdownV2")
        except Exception:
            logger.exception("MarkdownV2 standalone send failed for chunk %s", index + 1)
            fallback = plain_chunks[index] if index < len(plain_chunks) else chunk
            await _telegram_call_with_retry(chat.send_message, fallback)


async def run_scheduled_group_summary(
    state: BotState,
    chat_id: int,
    date_key: str,
) -> bool:
    """Generate and send one scheduled daily group summary.

    This function is intentionally independent of incoming messages. The main
    scheduler calls it around the configured local time, so a quiet group at
    23:00 can still be processed. Messages are cleared only after the summary
    has been successfully sent.
    """
    if chat_id in state.summary_in_progress:
        return False

    state.summary_in_progress.add(chat_id)
    try:
        rows = state.db.get_group_messages(chat_id, limit=120)
        if len(rows) < MIN_SUMMARY_MESSAGES:
            return False

        lines: list[ChatLine] = []
        for row in rows:
            timestamp = None
            raw_timestamp = row.get("created_at")
            if raw_timestamp:
                try:
                    timestamp = datetime.fromisoformat(raw_timestamp)
                except (TypeError, ValueError):
                    timestamp = None
            lines.append(
                ChatLine(
                    role="user",
                    text=str(row.get("text") or ""),
                    author=str(row.get("author") or "用户"),
                    message_id=int(row["message_id"]) if row.get("message_id") is not None else None,
                    chat_id=int(row["chat_id"]) if row.get("chat_id") is not None else chat_id,
                    chat_username=row.get("chat_username"),
                    timestamp=timestamp,
                )
            )

        if state.application is None:
            return False

        # Scheduled summaries should use the bot's configured default model,
        # because there is no single triggering user at 23:00.
        result = await summarize_chat(
            state.ai,
            lines,
            state.settings.default_model,
            automatic=True,
        )
        chat = await state.application.bot.get_chat(chat_id)
        await _send_markdown_to_chat(chat, result)

        last_message_id = max((line.message_id or 0) for line in lines)
        if last_message_id:
            if not state.db.clear_group_messages(chat_id, through_message_id=last_message_id):
                logger.error(
                    "Scheduled summary sent but failed to clear summarized messages for chat %s",
                    chat_id,
                )
                return False
        if not state.db.mark_summary_sent(chat_id, date_key):
            logger.error("Failed to mark scheduled summary as sent for chat %s", chat_id)
            return False
        state.clear_group_history_through(chat_id, last_message_id)
        return True
    except Exception:
        logger.exception("Scheduled group summary failed for chat %s", chat_id)
        return False
    finally:
        state.summary_in_progress.discard(chat_id)


async def summary_auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE, args: list[str]) -> None:
    state = get_state(context)
    chat = update.effective_chat
    if not chat or chat.type == ChatType.PRIVATE:
        await update.effective_message.reply_text("每日自动群总结只支持群聊。")
        return

    current = state.db.get_group_settings(chat.id)
    if not args:
        status = "开启" if current.auto_summary_enabled else "关闭"
        await update.effective_message.reply_text(
            f"📝 每日自动群总结：{status}\n"
            f"时间：{current.auto_summary_time}\n"
            f"时区：{current.timezone}\n"
            "每天到点后会自动总结当前已记录的有效群聊消息。"
        )
        return

    if not await _is_group_admin(update, context):
        await update.effective_message.reply_text("只有群管理员可以修改每日自动总结设置。")
        return

    token = args[0].lower()
    if token in {"on", "off"}:
        enabled = token == "on"
        new = GroupSettings(
            chat_id=chat.id,
            auto_history_enabled=current.auto_history_enabled,
            auto_history_time=current.auto_history_time,
            timezone=current.timezone,
            last_history_sent=current.last_history_sent,
            auto_summary_enabled=enabled,
            auto_summary_time=current.auto_summary_time,
            last_summary_sent=current.last_summary_sent,
        )
        if state.db.save_group_settings(new):
            await update.effective_message.reply_text(
                f"✅ 每日自动群总结已{'开启' if enabled else '关闭'}。时间：{new.auto_summary_time}（{new.timezone}）"
            )
        else:
            await update.effective_message.reply_text("⚠️ 保存自动总结设置失败。")
        return

    parsed_time = parse_time(token)
    if parsed_time:
        new = GroupSettings(
            chat_id=chat.id,
            auto_history_enabled=current.auto_history_enabled,
            auto_history_time=current.auto_history_time,
            timezone=current.timezone,
            last_history_sent=current.last_history_sent,
            auto_summary_enabled=True,
            auto_summary_time=parsed_time,
            last_summary_sent=current.last_summary_sent,
        )
        if state.db.save_group_settings(new):
            await update.effective_message.reply_text(
                f"✅ 每日自动群总结时间已设置为 {parsed_time}（{new.timezone}），并已开启。"
            )
        else:
            await update.effective_message.reply_text("⚠️ 保存自动总结设置失败。")
        return

    await update.effective_message.reply_text(
        "用法：/summary auto、/summary auto on、/summary auto off 或 /summary auto 23:00"
    )


async def summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE:
        await update.effective_message.reply_text("/summary 主要用于群聊。")
        return
    if context.args and context.args[0].lower() == "auto":
        await summary_auto_command(update, context, context.args[1:])
        return
    state = get_state(context)
    chat_lines = await _load_summary_lines(state, update.effective_chat.id)
    if len(chat_lines) < MIN_SUMMARY_MESSAGES:
        await update.effective_message.reply_text(
            f"📝 目前只有 {len(chat_lines)} 条有效消息，至少需要 {MIN_SUMMARY_MESSAGES} 条才能总结。"
        )
        return
    status = await update.effective_message.reply_text("📝 正在整理群聊总结……")
    model = state.db.get_user_model(update.effective_user.id, state.settings.default_model)
    try:
        result = await summarize_chat(state.ai, chat_lines, model, automatic=False)
        chunks = split_text(markdown_to_markdown_v2(result))
        try:
            await status.edit_text(chunks[0], parse_mode="MarkdownV2")
        except Exception:
            await status.edit_text(result.splitlines()[0] if result else "📝 总结完成")
            if result:
                chunks = []
        for chunk in chunks[1:]:
            await update.effective_chat.send_message(chunk, parse_mode="MarkdownV2")
    except Exception:
        logger.exception("Summary failed")
        await status.edit_text("⚠️ 当前无法完成群聊总结，请稍后重试。")


def _parse_history_arg(raw: str) -> tuple[int, int, int | None] | None:
    raw = raw.strip()
    if not raw:
        d = date.today()
        return d.month, d.day, None
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if iso:
        year, month, day = map(int, iso.groups())
        try:
            date(year, month, day)
        except ValueError:
            return None
        return month, day, year
    zh = re.fullmatch(r"(?:(\d{4})年)?\s*(\d{1,2})月\s*(\d{1,2})日", raw)
    if zh:
        year = int(zh.group(1)) if zh.group(1) else None
        month, day = int(zh.group(2)), int(zh.group(3))
        try:
            date(year or 2024, month, day)
        except ValueError:
            return None
        return month, day, year
    return None


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(context)
    args = context.args
    if args and args[0].lower() == "auto":
        await history_auto_command(update, context, args[1:])
        return
    if args and args[0].lower() == "timezone":
        await history_timezone_command(update, context, args[1:])
        return
    query = " ".join(args)
    if not query:
        local_today = __import__("datetime").datetime.now(ZoneInfo(state.settings.default_timezone)).date()
        parsed = (local_today.month, local_today.day, None)
    else:
        parsed = _parse_history_arg(query)
    if not parsed:
        await update.effective_message.reply_text("用法：/history、/history 8月8日、/history 2008-08-08")
        return
    month, day, year = parsed
    from .main import fetch_history_events, render_history_events
    events = await fetch_history_events(month, day, year, state.settings.history_language)
    if not events:
        await update.effective_message.reply_text("📅 暂时无法获取这一天的可靠历史资料，请稍后再试。")
        return
    await send_markdown(update, render_history_events(month, day, year, events))


async def _is_group_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE or not update.effective_user:
        return False
    try:
        member = await context.bot.get_chat_member(update.effective_chat.id, update.effective_user.id)
        return member.status in {"creator", "administrator"}
    except Exception:
        logger.exception("Failed to check administrator status")
        return False


async def history_auto_command(update: Update, context: ContextTypes.DEFAULT_TYPE, args: list[str]) -> None:
    state = get_state(context)
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE:
        await update.effective_message.reply_text("自动历史推送只支持群聊。")
        return
    current = state.db.get_group_settings(update.effective_chat.id)
    if not args:
        status = "开启" if current.auto_history_enabled else "关闭"
        await update.effective_message.reply_text(
            f"📅 自动历史推送：{status}\n时间：{current.auto_history_time}\n时区：{current.timezone}"
        )
        return
    if not await _is_group_admin(update, context):
        await update.effective_message.reply_text("只有群管理员可以修改自动推送设置。")
        return
    token = args[0].lower()
    if token in {"on", "off"}:
        enabled = token == "on"
        new = GroupSettings(
            chat_id=update.effective_chat.id,
            auto_history_enabled=enabled,
            auto_history_time=current.auto_history_time,
            timezone=current.timezone,
            last_history_sent=current.last_history_sent,
            auto_summary_enabled=current.auto_summary_enabled,
            auto_summary_time=current.auto_summary_time,
            last_summary_sent=current.last_summary_sent,
        )
        if state.db.save_group_settings(new):
            await update.effective_message.reply_text(f"✅ 自动历史推送已{'开启' if enabled else '关闭'}。")
        else:
            await update.effective_message.reply_text("⚠️ 保存设置失败。")
        return
    parsed_time = parse_time(token)
    if parsed_time:
        new = GroupSettings(
            chat_id=update.effective_chat.id,
            auto_history_enabled=True,
            auto_history_time=parsed_time,
            timezone=current.timezone,
            last_history_sent=current.last_history_sent,
            auto_summary_enabled=current.auto_summary_enabled,
            auto_summary_time=current.auto_summary_time,
            last_summary_sent=current.last_summary_sent,
        )
        if state.db.save_group_settings(new):
            await update.effective_message.reply_text(f"✅ 自动历史推送时间已设置为 {parsed_time}。")
        else:
            await update.effective_message.reply_text("⚠️ 保存设置失败。")
        return
    await update.effective_message.reply_text("用法：/history auto on|off 或 /history auto 08:00")


async def history_timezone_command(update: Update, context: ContextTypes.DEFAULT_TYPE, args: list[str]) -> None:
    state = get_state(context)
    if not update.effective_chat or update.effective_chat.type == ChatType.PRIVATE:
        await update.effective_message.reply_text("时区设置只支持群聊。")
        return
    if not await _is_group_admin(update, context):
        await update.effective_message.reply_text("只有群管理员可以修改时区。")
        return
    if not args or not valid_timezone(args[0]):
        await update.effective_message.reply_text("请输入有效 IANA 时区，例如 Asia/Shanghai。")
        return
    current = state.db.get_group_settings(update.effective_chat.id)
    new = GroupSettings(
        chat_id=update.effective_chat.id,
        auto_history_enabled=current.auto_history_enabled,
        auto_history_time=current.auto_history_time,
        timezone=args[0],
        last_history_sent=current.last_history_sent,
        auto_summary_enabled=current.auto_summary_enabled,
        auto_summary_time=current.auto_summary_time,
        last_summary_sent=current.last_summary_sent,
    )
    if state.db.save_group_settings(new):
        await update.effective_message.reply_text(f"✅ 群时区已设置为 {args[0]}。")
    else:
        await update.effective_message.reply_text("⚠️ 保存时区失败。")


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(context)
    model = state.db.get_user_model(update.effective_user.id, state.settings.default_model)
    await update.effective_message.reply_text(
        f"✅ Bot V2 正常运行\nAI：OpenAI-compatible\n模型：{model}\n搜索：DDGS + 可选 SearXNG\nWebhook：启用"
    )


async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    start = time.perf_counter()
    sent = await update.effective_message.reply_text("🏓 Pong")
    elapsed = round((time.perf_counter() - start) * 1000)
    await sent.edit_text(f"🏓 Pong\n响应时间：{elapsed} ms")


async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = get_state(context)
    model = state.db.get_user_model(update.effective_user.id, state.settings.default_model)
    await update.effective_message.reply_text(
        f"Telegram AI Bot V2\n版本：2.0\nAI Provider：OpenAI-compatible\n当前模型：{model}\n联网搜索：DDGS\n历史数据：Wikimedia / Wikipedia 数据源"
    )


async def _extract_user_prompt(update: Update, state: BotState) -> tuple[str, ReplyContext, Any | None]:
    message = update.effective_message
    assert message
    prompt = normalize_text(message.caption or message.text or "")
    reply_context = build_reply_context(message, state.settings.max_reply_context_length)
    target_image = None
    if message.photo:
        target_image = message
    elif reply_context.replied_has_image and message.reply_to_message:
        target_image = message.reply_to_message
    if message.chat.type != ChatType.PRIVATE and state.bot_username:
        pattern = rf"@{re.escape(state.bot_username)}\b"
        prompt = re.sub(pattern, "", prompt, flags=re.IGNORECASE).strip()
    return truncate_text(prompt, state.settings.max_input_chars), reply_context, target_image


def should_handle_group_message(update: Update, state: BotState) -> bool:
    message = update.effective_message
    if not message or not update.effective_chat:
        return False
    if update.effective_chat.type == ChatType.PRIVATE:
        return True
    text = message.text or message.caption or ""
    mentioned = bool(
        state.bot_username
        and re.search(rf"@{re.escape(state.bot_username)}\b", text, flags=re.IGNORECASE)
    )
    replied_to_bot = bool(
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.username
        and state.bot_username
        and message.reply_to_message.from_user.username.lower() == state.bot_username.lower()
    )
    return mentioned or replied_to_bot


async def _send_ai_answer(update: Update, answer: str) -> None:
    if not update.effective_message:
        return
    thinking = await _telegram_call_with_retry(update.effective_message.reply_text, "🤔")
    rendered_chunks = split_text(markdown_to_markdown_v2(answer))
    plain_chunks = split_text(answer)
    if not rendered_chunks:
        await _telegram_call_with_retry(thinking.edit_text, "（AI 返回了空消息）")
        return

    try:
        await _telegram_call_with_retry(thinking.edit_text, rendered_chunks[0], parse_mode="MarkdownV2")
    except Exception:
        logger.exception("Failed to render AI answer as MarkdownV2")
        fallback = plain_chunks[0] if plain_chunks else rendered_chunks[0]
        await _telegram_call_with_retry(thinking.edit_text, fallback)

    for index, chunk in enumerate(rendered_chunks[1:], start=1):
        try:
            await _telegram_call_with_retry(
                update.effective_message.chat.send_message,
                chunk,
                parse_mode="MarkdownV2",
            )
        except Exception:
            logger.exception("Failed to send AI answer chunk %s as MarkdownV2", index + 1)
            fallback_index = index if index < len(plain_chunks) else len(plain_chunks) - 1
            fallback = plain_chunks[fallback_index] if plain_chunks and fallback_index >= 0 else chunk
            await _telegram_call_with_retry(update.effective_message.chat.send_message, fallback)


async def _record_group_message(update: Update, state: BotState) -> bool:
    chat = update.effective_chat
    message = update.effective_message
    if not chat or not message or chat.type == ChatType.PRIVATE:
        return False
    raw = normalize_text(message.text or message.caption or "")
    if not raw or raw.lstrip().startswith("/") or not _is_meaningful_group_message(raw):
        return False
    author = (update.effective_user.full_name if update.effective_user else "用户") or "用户"
    timestamp = message.date.isoformat() if message.date else None
    # Touch the group settings row so the daily scheduler can discover this group
    # even when nobody has configured the /history feature.
    state.db.ensure_group_settings(chat.id)
    state.add_group_message(
        chat.id,
        author,
        raw,
        message_id=message.message_id,
        chat_username=chat.username,
        timestamp=message.date,
    )
    state.db.save_group_message(
        chat.id, message.message_id, author, truncate_text(raw, 4000),
        chat_username=chat.username, created_at=timestamp,
    )
    return True


async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.effective_message or not update.effective_chat:
        return
    state = get_state(context)
    group_chat = update.effective_chat.type != ChatType.PRIVATE

    # Serialize all work inside one chat. Different chats are still handled
    # concurrently by PTB, but shared group context cannot race or reorder.
    async with state.lock_for(update):
        if group_chat:
            # A failed DB write must never prevent the actual AI response.
            try:
                await _record_group_message(update, state)
            except Exception:
                logger.exception("Failed to persist group message; continuing with AI handling")

        if not should_handle_group_message(update, state):
            return

        try:
            try:
                await _telegram_call_with_retry(
                    update.effective_chat.send_action,
                    ChatAction.TYPING,
                    attempts=2,
                )
            except Exception:
                logger.warning("Unable to send typing action", exc_info=True)

            prompt, reply_context, image_message = await _extract_user_prompt(update, state)
            if reply_context.quoted_text:
                prompt_for_ai = (
                    f"明确引用内容（优先上下文）：\n{reply_context.quoted_text}"
                    f"\n\n当前问题：\n{prompt}"
                )
            elif reply_context.replied_text:
                prompt_for_ai = (
                    f"被回复消息（优先上下文）：\n{reply_context.replied_text}"
                    f"\n\n当前问题：\n{prompt}"
                )
            else:
                prompt_for_ai = prompt

            if not prompt_for_ai and image_message is None:
                return

            user_id = update.effective_user.id if update.effective_user else 0
            model = state.db.get_user_model(user_id, state.settings.default_model)

            if image_message is not None:
                answer = await asyncio.wait_for(
                    analyze_telegram_image(state.ai, image_message, prompt, model),
                    timeout=90.0,
                )
            else:
                messages: list[dict[str, str]] = [
                    {
                        "role": "system",
                        "content": (
                            "你是 Telegram AI 助手。回答要准确、自然、使用 Markdown。"
                            "不要输出 <think>、<thinking>、<analysis>、<reasoning> 标签或内部推理过程，只输出最终答案。"
                        ),
                    }
                ]
                if group_chat:
                    messages.append({
                        "role": "system",
                        "content": (
                            "这是共享的群聊短期上下文。不同成员的普通消息都可能是当前问题的背景。"
                            "不要假设提问者就是这些消息的作者；按消息中的作者前缀理解。"
                            "如果当前问题明确引用/回复某条消息，应优先理解那条直接引用。"
                        ),
                    })
                previous_context = state.recent_context(update)
                if group_chat and previous_context:
                    previous_context = previous_context[:-1]
                messages.extend(previous_context)
                messages.append({
                    "role": "user",
                    "content": truncate_text(prompt_for_ai, state.settings.context_max_chars),
                })
                answer = await asyncio.wait_for(
                    state.ai.chat(messages, model=model, max_tokens=3500),
                    timeout=65.0,
                )

            answer = truncate_text(answer, state.settings.max_output_chars)
            await asyncio.wait_for(_send_ai_answer(update, answer), timeout=30.0)

            if group_chat:
                state.add_group_assistant(update.effective_chat.id, answer)
                state.mark_bot_activity(update.effective_chat.id)
            else:
                state.add_context(update, prompt_for_ai, answer)
        except asyncio.TimeoutError:
            await _telegram_call_with_retry(
                update.effective_message.reply_text,
                "⚠️ 本次处理超时了，请稍后重新发送。",
            )
        except VisionError as exc:
            await _telegram_call_with_retry(
                update.effective_message.reply_text,
                f"⚠️ 图片处理失败：{exc}",
            )
        except AIError:
            await _telegram_call_with_retry(
                update.effective_message.reply_text,
                "⚠️ AI 服务暂时不可用，请稍后重试。",
            )
        except Exception:
            logger.exception("Message handler failed")
            try:
                await _telegram_call_with_retry(
                    update.effective_message.reply_text,
                    "⚠️ 处理消息时出现问题，但 Bot 本身仍在运行。",
                )
            except Exception:
                logger.exception("Failed to send message-handler error notice")

