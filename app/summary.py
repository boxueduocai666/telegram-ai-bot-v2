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
MAX_SUMMARY_TOPICS = 5


SUMMARY_SYSTEM_PROMPT = r"""你是一个长期运行的 Telegram 群聊总结助手。

你的任务不是逐条复述聊天记录，也不是给每条消息分类，而是像一名真正的编辑一样，从整段聊天中提炼“大家到底在聊什么、表达了什么、最后形成了什么结论”。

旧版 Telegram AI Bot 的总结风格是参考标准：自然、概括性强、有语义归纳，并在最后给出“总体结论”。新版 V2 还需要保留时间范围、有效消息数量和代表消息跳转能力。

请输出 JSON，不要输出 Markdown 代码块，不要输出额外解释：
{
  "topics": [
    {
      "title": "自然、概括性的主题标题",
      "summary": "对该主题做自然总结，说明主要内容、发展或表达；不要只是复述消息表面文字",
      "message_indexes": [1]
    }
  ],
  "overall_conclusion": "对整段群聊的性质、主要内容、参与情况和整体氛围做简洁判断"
}

总结原则：
1. 先理解整段聊天，再提炼主题。不要一句消息一个主题，也不要把“连续发了很多消息”本身当成主题。
2. 自动过滤低信息量内容，例如打招呼、哈哈、6、好的、收到、单个表情等。
3. 对零散、无明显语义关联的测试词、误触词或碎片，不要强行包装成“主要话题”。除非它们确实形成了可识别的讨论。
4. 一个主题应描述“这段聊天在讨论/表达什么”，而不是简单描述消息类型。例如不要写“古风诗词连发”，更推荐“古诗词与诗意表达”或“诗词中的情感抒发”。
5. 将连续发送、同一人物的多条相关内容合并理解；不要机械拆分。
6. 话题数量按实际内容决定，通常 1~5 个。绝对不要为了凑数量而制造主题。
7. 如果整段聊天只有一个核心主题，就只输出一个主题；如果聊天内容非常零散，也可以少写甚至没有主题。
8. 每个主题通常用 1~3 句说明，重点写“发生了什么、表达了什么、讨论走向如何”。可以保留少量有代表性的原话，但不要堆砌原文。
9. 不要编造人物关系、态度、因果、结论或聊天中没有出现的事实。
10. 若主要只有一个人在连续分享内容，应在 overall_conclusion 中如实指出“主要由单一用户发起/分享”，但不要贬低或嘲讽内容。
11. 如果没有形成真正的多人讨论、问答或互动，应明确说明这一点。这是有效的总结判断，不需要强行制造“讨论结果”。
12. overall_conclusion 要像 V1 的“总体结论”：对整段聊天做整体判断，而不是再次列一遍话题。
13. 每个主题至少选择 1 条最有代表性的 message_indexes，编号必须真实存在；优先选择 1~3 条关键消息。
14. 不要输出 URL、相关消息区域、MSG 编号文本、Markdown 或 emoji。Bot 会根据 message_indexes 自动生成可点击的主题标题。
15. 只使用输入中真实存在的消息。
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


def _normalize_topic_indexes(topic: dict, line_count: int) -> list[int]:
    raw = topic.get("message_indexes") or []
    if not isinstance(raw, list):
        raw = [raw]
    result: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= line_count and index not in result:
            result.append(index)
    return result[:3]


def _fallback_summary(lines: Sequence[ChatLine], automatic: bool = False) -> str:
    title = "📝 **群聊自动总结**" if automatic else "📝 **群聊 AI 总结**"
    if not lines:
        return f"{title}\n\n暂时没有足够的有效聊天内容可以总结。"

    parts = [title]
    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 时间范围：{time_range}")
    parts.append(f"\n💬 已分析 {len(lines)} 条有效消息")

    parts.append("\n\n📌 当前聊天内容")
    for line in lines[:4]:
        text = truncate_text(line.text, 180)
        if text:
            parts.append(f"\n{line.author or '用户'}：{text}")

    authors = {line.author for line in lines if line.author}
    if len(authors) <= 1:
        conclusion = "本次聊天主要由单一用户发送内容，暂未形成明显的多人讨论。"
    else:
        conclusion = "当前聊天内容较少，暂时无法提炼出更明确的讨论结论。"
    parts.append(f"\n\n💡 **总体结论**\n{conclusion}")
    return "".join(parts)


def render_summary(data: dict, lines: Sequence[ChatLine], *, automatic: bool = False) -> str:
    title = "📝 **群聊自动总结**" if automatic else "📝 **群聊 AI 总结**"
    topics = data.get("topics") or []
    if not isinstance(topics, list):
        topics = []

    parts = [title]
    time_range = _format_time_range(lines)
    if time_range:
        parts.append(f"\n🕒 时间范围：{time_range}")

    # 先过滤掉 AI 返回的无效主题，避免“凑数量”。
    valid_topics: list[tuple[str, str, list[int]]] = []
    for topic in topics:
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
    if conclusion:
        parts.append(f"\n\n💡 **总体结论**\n{conclusion}")
    elif valid_topics:
        authors = {line.author for line in lines if line.author}
        if len(authors) <= 1:
            conclusion = "本次聊天主要由单一用户发起和分享，未形成明显的多人讨论或互动。"
        else:
            conclusion = "本次聊天围绕上述主题展开，整体未形成更明确的统一结论。"
        parts.append(f"\n\n💡 **总体结论**\n{conclusion}")

    if not valid_topics:
        return _fallback_summary(lines, automatic=automatic)
    return "".join(parts)


async def summarize_chat(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> str:
    valid_lines = [line for line in lines if line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        title = "📝 **群聊自动总结**" if automatic else "📝 **群聊 AI 总结**"
        return (
            f"{title}\n\n"
            f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条有效消息。"
        )

    content = "\n".join(
        f"[MSG:{index}] 用户：{line.author or '用户'}\n内容：{line.text}"
        for index, line in enumerate(valid_lines, start=1)
    )
    content = truncate_text(content, 30000)

    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "以下是需要总结的 Telegram 群聊记录。请先理解整个聊天的语义关系，"
                        "再按照系统要求提炼主题和总体结论。不要为了覆盖每条消息而强行增加主题。\n\n"
                        + content
                    ),
                },
            ],
            model=model,
            temperature=0.2,
            max_tokens=5000,
        )
        data = _parse_summary_json(raw)
        return render_summary(data, valid_lines, automatic=automatic) if data else _fallback_summary(
            valid_lines, automatic=automatic
        )
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
