from __future__ import annotations

import json
from dataclasses import dataclass
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


MIN_SUMMARY_MESSAGES = 3

SUMMARY_SYSTEM_PROMPT = """你是一个有温度、自然、简洁的 Telegram 群聊总结助手。
请根据聊天记录生成“群聊 AI 总结”，风格参考 Telegram 群里的轻量日报，而不是严肃会议纪要。

要求：
- 只总结聊天中真实出现的内容，不补充不存在的事实。
- 只输出 JSON，不要输出 Markdown 代码块，不要输出额外解释。
- JSON 格式必须为：
{
  "topics": [
    {"title": "主题标题", "summary": "1~2句自然描述", "message_indexes": [1, 2]}
  ],
  "conclusion": "总体结论"
}
- topics 按讨论重要程度排序，通常 1~5 个即可。
- message_indexes 使用提供的聊天记录编号，从 1 开始；每个主题至少选择 1 条最能代表该主题的原始消息。
- 如果聊天内容不足以形成多个主题，就少输出几个主题，不要硬凑栏目。
- 不要输出“已确定事项”“待解决问题”等固定栏目，除非它们确实是自然且重要的内容；本格式不需要这些栏目。
- 标题要短、自然，适合直接显示在 Telegram 消息里。
- 可以合理使用少量 Emoji，让结果有 Telegram 群聊日报的感觉；不要每句话都加 Emoji。
- 可以使用 🤔、💬、🔥、📌 等自然的 Emoji，但不要为了凑表情强行添加。
"""


def _fallback_summary(lines: Sequence[ChatLine]) -> str:
    """Readable V1-style fallback when the model does not return valid JSON."""
    if not lines:
        return "📝 群聊 AI 总结\n\n💡 总体结论\n暂时没有足够的聊天内容可以总结。"
    preview = "；".join(
        f"{line.author or '用户'}：{truncate_text(line.text, 80)}" for line in lines[:3]
    )
    return f"📝 群聊 AI 总结\n\n📌 最近讨论\n{preview}\n\n💡 总体结论\n聊天内容较少，暂未形成明确的讨论主题。"


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
    # Telegram private/supergroup message links use the internal chat id without -100.
    chat_id = str(line.chat_id)
    if chat_id.startswith("-100"):
        return f"https://t.me/c/{chat_id[4:]}/{line.message_id}"
    return None


def render_summary(data: dict, lines: Sequence[ChatLine]) -> str:
    parts = ["📝 **群聊 AI 总结**"]
    topics = data.get("topics") or []
    used_indexes: set[int] = set()

    for topic in topics:
        if not isinstance(topic, dict):
            continue
        title = str(topic.get("title") or "最近讨论").strip()
        description = str(topic.get("summary") or "").strip()
        indexes = topic.get("message_indexes") or []
        valid_indexes = [i for i in indexes if isinstance(i, int) and 1 <= i <= len(lines)]
        if not title or not description:
            continue
        link = None
        for idx in valid_indexes:
            link = _telegram_message_url(lines[idx - 1])
            if link:
                used_indexes.add(idx)
                break
        if link:
            # The title itself is the jump target; no ugly URL is printed below.
            parts.append(f"\n📌 [{title}]({link})\n{description}")
        else:
            parts.append(f"\n📌 {title}\n{description}")

    conclusion = str(data.get("conclusion") or "").strip()
    if conclusion:
        parts.append(f"\n💡 **总体结论**\n{conclusion}")

    if len(parts) == 1:
        return _fallback_summary(lines)
    return "".join(parts)


async def summarize_chat(ai: AIClient, lines: Sequence[ChatLine], model: str) -> str:
    valid_lines = [line for line in lines if line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条消息。"

    content = "\n".join(
        f"[{index}] {line.author + ': ' if line.author else ''}{line.text}"
        for index, line in enumerate(valid_lines, start=1)
    )
    content = truncate_text(content, 20000)
    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
        )
        data = _parse_summary_json(raw)
        if data is None:
            return _fallback_summary(valid_lines)
        return render_summary(data, valid_lines)
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
