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


SUMMARY_SYSTEM_PROMPT = """你是 Telegram 群聊 AI 总结助手。
请把提供的群聊内容整理成一种轻松、自然、好读的“群聊 AI 总结”，风格严格参考下面这种感觉：

📝 群聊 AI 总结

📌 一个讨论主题
用一两句自然的中文说明发生了什么，不要写成会议纪要。

📌 另一个讨论主题
用一两句自然的中文说明发生了什么。

💡 总体结论
用一两句概括今天群聊的主要方向和讨论程度。

要求：
- 只总结聊天中真实出现的内容，不补充不存在的事实。
- 不要使用“1. 最近讨论主题 / 2. 主要观点 / 3. 已确定事项 / 4. 待解决问题”这种固定报告模板。
- 有几个真正值得总结的话题就写几个，没有就少写，不要硬凑栏目。
- 每个主题标题要简短、自然，适合直接显示在 Telegram 中。
- 描述保持简洁，通常 1~2 句即可。
- 必须有“总体结论”，但不要重复前面的内容。
- 可以自然使用少量 Emoji，例如 📌、💡、🤔、💬、🔥、❓、✅；不要为了凑数量强行添加。
- 输出 JSON，不要输出 Markdown 代码块或任何额外解释。
- JSON 格式必须为：
{
  "topics": [
    {"title": "主题标题", "summary": "自然描述", "message_indexes": [1]}
  ],
  "conclusion": "总体结论"
}
- message_indexes 是最能代表该主题的原始消息编号，从 1 开始；至少选择 1 条。
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


def _fallback_summary(lines: Sequence[ChatLine]) -> str:
    """Keep the V1 visual style if the model returns malformed JSON."""
    if not lines:
        return "📝 群聊 AI 总结\n\n💡 总体结论\n暂时没有足够的聊天内容可以总结。"

    parts = ["📝 群聊 AI 总结"]
    for line in lines[:5]:
        text = truncate_text(line.text, 100)
        if text:
            parts.append(f"\n\n📌 {text}")
    parts.append("\n\n💡 总体结论\n群聊内容较少，暂时以话题引入和简短交流为主。")
    return "".join(parts)


def render_summary(data: dict, lines: Sequence[ChatLine]) -> str:
    """Render the V1-style summary; topic titles themselves are clickable."""
    parts = ["📝 **群聊 AI 总结**"]
    topics = data.get("topics") or []

    for topic in topics:
        if not isinstance(topic, dict):
            continue
        title = str(topic.get("title") or "").strip()
        description = str(topic.get("summary") or "").strip()
        if not title or not description:
            continue

        indexes = topic.get("message_indexes") or []
        link = None
        if isinstance(indexes, list):
            for index in indexes:
                if isinstance(index, int) and 1 <= index <= len(lines):
                    link = _telegram_message_url(lines[index - 1])
                    if link:
                        break

        if link:
            parts.append(f"\n\n📌 [{title}]({link})\n{description}")
        else:
            parts.append(f"\n\n📌 {title}\n{description}")

    conclusion = str(data.get("conclusion") or "").strip()
    if conclusion:
        parts.append(f"\n\n💡 **总体结论**\n{conclusion}")

    return "".join(parts) if len(parts) > 1 else _fallback_summary(lines)


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
        return render_summary(data, valid_lines) if data else _fallback_summary(valid_lines)
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
