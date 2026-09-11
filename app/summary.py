from __future__ import annotations

import json
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

SUMMARY_SYSTEM_PROMPT = """你是一个长期运行的 Telegram 群聊总结助手。
请把提供的群聊整理成一种像“群聊日报/新闻速览”一样自然、具体、有信息量的总结，而不是会议纪要。

输出 JSON，不要输出 Markdown 代码块或任何额外解释。
JSON 格式：
{
  "topics": [
    {
      "title": "简短自然的主题标题",
      "summary": "具体说明发生了什么、谁参与、事情如何发展；有原话时可自然保留少量原话",
      "message_indexes": [1]
    }
  ]
}

要求：
- 只总结聊天里真实出现的内容，不编造时间、人物、结论、因果或态度。
- 自动忽略“哈哈”“6”“好的”“收到”、单独表情、纯打招呼等低信息量消息。
- 优先识别真正发生了变化或形成讨论的事件：新闻、问题、争论、产品、考试、活动、故障、计划、链接分享等。
- 话题数量按实际内容决定：通常 3~7 个；聊天少就少写，内容多就不要只塞 3 个。
- 每个话题通常写 2~4 句，让读者看得出“发生了什么”和“讨论怎么发展”。不要机械套“观点/结论/待办”模板。
- 人名可以直接使用消息里的显示名；不要擅自给人起外号。
- 可以适量保留有代表性的原话，让总结有群聊味道，但不要堆砌原文。
- 每个话题至少选择 1 条最有代表性的 message_indexes；这些编号必须真实存在。
- 一个话题可以选择多个 message_indexes，但优先选择 1~3 条最关键的。
- 不要输出 URL、不要生成“相关消息”区域；Bot 会根据 message_indexes 自动生成可点击跳转。
- 不要输出“总体结论”之类的固定尾巴，除非聊天本身确实出现了明确的整体结论。
"""


def _parse_summary_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("topics"), list):
        return None
    return data


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
    start = min(times)
    end = max(times)
    return f"{start:%m-%d %H:%M} — {end:%m-%d %H:%M}"


def _fallback_summary(lines: Sequence[ChatLine], automatic: bool = False) -> str:
    if not lines:
        return "📝 群聊 AI 总结\n\n暂时没有足够的有效聊天内容可以总结。"
    title = "📝 **群聊 AI 总结（自动）**" if automatic else "📝 **群聊 AI 总结**"
    parts = [title]
    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 {time_range}")
    parts.append(f"\n💬 已分析 {len(lines)} 条有效消息")
    for index, line in enumerate(lines[:5], start=1):
        text = truncate_text(line.text, 140)
        if text:
            parts.append(f"\n\n{index}. {text}")
    return "".join(parts)


def render_summary(data: dict, lines: Sequence[ChatLine], *, automatic: bool = False) -> str:
    title = "📝 **群聊 AI 总结（自动）**" if automatic else "📝 **群聊 AI 总结**"
    topics = data.get("topics") or []
    parts = [title]

    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 {time_range}")
    parts.append(f"\n💬 已分析 {len(lines)} 条有效消息，整理出 {len(topics)} 个主要话题")

    rendered_count = 0
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        title_text = str(topic.get("title") or "").strip()
        description = str(topic.get("summary") or "").strip()
        if not title_text or not description:
            continue

        rendered_count += 1
        indexes = topic.get("message_indexes") or []
        link = None
        if isinstance(indexes, list):
            for index in indexes:
                if isinstance(index, int) and 1 <= index <= len(lines):
                    link = _telegram_message_url(lines[index - 1])
                    if link:
                        break

        if link:
            parts.append(f"\n\n{rendered_count}. [{title_text}]({link})\n{description}")
        else:
            parts.append(f"\n\n{rendered_count}. {title_text}\n{description}")

    return "".join(parts) if rendered_count else _fallback_summary(lines, automatic=automatic)


async def summarize_chat(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> str:
    valid_lines = [line for line in lines if line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条有效消息。"

    content = "\n".join(
        f"[{index}] {line.author + ': ' if line.author else ''}{line.text}"
        for index, line in enumerate(valid_lines, start=1)
    )
    content = truncate_text(content, 30000)
    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
            max_tokens=5000,
        )
        data = _parse_summary_json(raw)
        return render_summary(data, valid_lines, automatic=automatic) if data else _fallback_summary(valid_lines, automatic=automatic)
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
