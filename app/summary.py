from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .ai import AIClient, AIError
from .utils import truncate_text


@dataclass(frozen=True)
class ChatLine:
    role: str
    text: str
    author: str = ""
    message_id: int | None = None
    chat_id: int | None = None
    chat_username: str | None = None
    timestamp: datetime | None = None


MIN_SUMMARY_MESSAGES = 3
MAX_SUMMARY_TOPICS = 5
SUMMARY_MAX_INPUT_CHARS = 30000

# Keep the same low-information filter used while recording new group messages.
# This also protects summaries built from messages saved by an older V2 version.
IGNORED_SUMMARY_TEXT = {
    "你好", "嗨", "哈喽", "hello", "hi", "早", "早上好", "晚上好", "晚安",
    "哈哈", "哈哈哈", "哈哈哈哈", "嗯", "哦", "噢", "啊", "好的", "好", "收到",
    "ok", "OK", "666", "6",
}

SUMMARY_SYSTEM_PROMPT = r"""你是一个长期运行的 Telegram 群聊总结助手。

请像真正的群聊编辑一样，总结整段聊天中“大家在聊什么、发生了什么、讨论如何发展”，而不是逐条复述。
输出必须是 JSON，不能输出 Markdown 代码块、解释文字或 JSON 之外的内容：
{
  "topics": [
    {
      "title": "自然、简洁的主题标题",
      "summary": "1~3句自然中文总结，说明主要内容、发展或表达；只使用聊天里真实出现的信息",
      "message_indexes": [1]
    }
  ],
  "overall_conclusion": "对整段聊天做简洁的总体判断"
}

规则：
1. 自动忽略打招呼、哈哈、6、好的、收到、单个表情等低信息量消息。
2. 不要一句消息一个主题；同一件事的连续消息要合并理解。
3. 只输出真正形成话题的内容。聊天很零散时可以只有1个主题，甚至没有主题。
4. 通常输出1~5个主题，不要为了凑数制造主题。
5. 主题标题要描述“在讨论/表达什么”，不要写成“某人连续发了几条消息”这种消息类型说明。
6. 不编造人物关系、态度、因果、结论或聊天中没有出现的事实。
7. 若主要由单一用户分享，应在总体结论中如实说明，但不要嘲讽。
8. 如果没有形成多人讨论、问答或明确结论，要如实写出这一点，不要强行制造结论。
9. 每个主题至少选择1条最有代表性的 message_indexes，编号必须来自输入消息，优先1~3条。
10. 不要输出 URL、MSG 文本、emoji、Markdown标记；Bot 会自动生成可点击的主题链接。
"""


def _parse_summary_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, count=1, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text, count=1)
        text = text.strip()

    # Be tolerant if an otherwise useful model response contains a tiny amount
    # of prose before/after the JSON object.
    candidates = [text]
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append(text[first : last + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("topics"), list):
            return data
    return None


def _telegram_message_url(line: ChatLine) -> str | None:
    if not line.message_id or not line.chat_id:
        return None
    if line.chat_username:
        username = line.chat_username.lstrip("@").strip()
        if username:
            return f"https://t.me/{username}/{line.message_id}"
    chat_id = str(line.chat_id)
    if chat_id.startswith("-100"):
        return f"https://t.me/c/{chat_id[4:]}/{line.message_id}"
    return None


def _format_time_range(lines: Sequence[ChatLine]) -> str | None:
    times = [line.timestamp for line in lines if line.timestamp]
    if not times:
        return None
    try:
        start = min(times)
        end = max(times)
    except TypeError:
        return None
    return f"{start:%m-%d %H:%M} — {end:%m-%d %H:%M}"


def _normalize_topic_indexes(topic: dict, line_count: int) -> list[int]:
    raw = topic.get("message_indexes") or []
    if not isinstance(raw, list):
        raw = [raw]
    indexes: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= line_count and index not in indexes:
            indexes.append(index)
    return indexes[:3]


def _is_summary_meaningful(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned or cleaned in IGNORED_SUMMARY_TEXT or len(cleaned) <= 1:
        return False
    if re.fullmatch(r"[\W_]+", cleaned, flags=re.UNICODE):
        return False
    return True


def _fallback_summary(lines: Sequence[ChatLine]) -> str:
    title = "📝 **群聊 AI 总结**"
    if not lines:
        return f"{title}\n\n暂时没有足够的有效聊天内容可以总结。"

    parts = [title]
    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 时间范围：{time_range}")
    parts.append(f"\n💬 已分析 {len(lines)} 条有效消息")
    parts.append("\n\n📌 当前聊天内容")
    for line in lines[:4]:
        snippet = truncate_text(line.text, 180)
        if snippet:
            parts.append(f"\n{line.author or '用户'}：{snippet}")

    authors = {line.author for line in lines if line.author}
    if len(authors) <= 1:
        conclusion = "本次聊天主要由单一用户发送或分享内容，暂未形成明显的多人讨论。"
    else:
        conclusion = "当前聊天内容较少，暂时无法提炼出更明确的统一讨论结论。"
    parts.append(f"\n\n💡 **总体结论**\n{conclusion}")
    return "".join(parts)


def render_summary(data: dict | None, lines: Sequence[ChatLine], *, automatic: bool = False) -> str:
    # The public format intentionally stays identical for manual and daily
    # summaries, so the group's visual experience does not regress between modes.
    title = "📝 **群聊 AI 总结**"
    if not data or not isinstance(data, dict):
        return _fallback_summary(lines)

    raw_topics = data.get("topics") or []
    valid_topics: list[tuple[str, str, list[int]]] = []
    if isinstance(raw_topics, list):
        for topic in raw_topics:
            if not isinstance(topic, dict):
                continue
            title_text = str(topic.get("title") or "").strip()
            description = str(topic.get("summary") or "").strip()
            indexes = _normalize_topic_indexes(topic, len(lines))
            if not title_text or not description or not indexes:
                continue
            valid_topics.append((title_text, description, indexes))
            if len(valid_topics) >= MAX_SUMMARY_TOPICS:
                break

    if not valid_topics:
        return _fallback_summary(lines)

    parts = [title]
    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 时间范围：{time_range}")
    parts.append(
        f"\n💬 已分析 {len(lines)} 条有效消息，整理出 {len(valid_topics)} 个主要话题"
    )

    for title_text, description, indexes in valid_topics:
        link = None
        for index in indexes:
            link = _telegram_message_url(lines[index - 1])
            if link:
                break
        if link:
            parts.append(f"\n\n[📌 {title_text}]({link})\n{description}")
        else:
            parts.append(f"\n\n📌 **{title_text}**\n{description}")

    conclusion = str(data.get("overall_conclusion") or "").strip()
    if not conclusion:
        authors = {line.author for line in lines if line.author}
        if len(authors) <= 1:
            conclusion = "本次聊天主要由单一用户发起和分享，未形成明显的多人讨论或互动。"
        else:
            conclusion = "本次聊天围绕上述主题展开，整体未形成更明确的统一结论。"
    parts.append(f"\n\n💡 **总体结论**\n{conclusion}")
    return "".join(parts)


async def summarize_chat(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> str:
    valid_lines = [line for line in lines if _is_summary_meaningful(line.text)]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return (
            "📝 **群聊 AI 总结**\n\n"
            f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条有效消息。"
        )

    content = "\n".join(
        f"[MSG:{index}] 用户：{line.author or '用户'}\n内容：{line.text}"
        for index, line in enumerate(valid_lines, start=1)
    )
    content = truncate_text(content, SUMMARY_MAX_INPUT_CHARS)

    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "以下是需要总结的 Telegram 群聊记录。请先理解整体语义关系，"
                        "再提炼真正的话题和总体结论。不要为了覆盖每条消息而强行增加主题。\n\n"
                        + content
                    ),
                },
            ],
            model=model,
            temperature=0.2,
            max_tokens=5000,
        )
        data = _parse_summary_json(raw)
        return render_summary(data, valid_lines, automatic=automatic)
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
