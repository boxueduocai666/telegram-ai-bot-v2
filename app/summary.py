from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecoder
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
    message_link: str | None = None


@dataclass(frozen=True)
class SummaryTopic:
    title: str
    description: str
    message_id: int | None = None
    message_link: str | None = None


@dataclass(frozen=True)
class ChatSummary:
    topics: tuple[SummaryTopic, ...]
    conclusion: str = ""


MIN_SUMMARY_MESSAGES = 3
MAX_TOPICS = 7
MAX_TOPIC_TITLE_LENGTH = 60
MAX_TOPIC_DESCRIPTION_LENGTH = 700


SUMMARY_SYSTEM_PROMPT = """你是 Telegram 群聊日报总结助手。

请根据提供的群聊消息，生成一份自然、具体、好读的群聊总结。
重点是还原真实讨论，不要编造事实，也不要为了凑栏目硬写内容。

严格输出 JSON，不要输出 Markdown 代码块，不要输出任何解释。格式：
{
  "topics": [
    {
      "title": "简短自然的主题标题",
      "description": "用 1-3 句话概括这个话题实际聊了什么",
      "message_id": 123
    }
  ],
  "conclusion": "用 1-2 句话概括整段聊天"
}

规则：
1. topics 只保留真正值得总结的讨论，通常 3-7 个，内容少就少写。
2. 每个 topic 只需要选择 1 个最有代表性的真实 message_id；只能使用输入里出现过的 ID。
3. description 只能依据聊天内容，不得补充不存在的人物、时间、因果或结论。
4. 忽略“哈哈”“6”“好的”“收到”、纯表情、单独打招呼等低信息量消息。
5. 每个话题写 1-3 句话，具体说明“发生了什么、讨论怎么发展”，不要写成机械模板。
6. 可以直接使用消息里的显示名，不要擅自改名。
7. 不要输出 URL，不要输出 message_indexes、message_ids 数组、相关消息区域或其他内部字段。
8. conclusion 必须是自然的整体概括；没有明确结论时，也可以写“本次聊天主要围绕……展开”。
9. 不要输出“主要观点/已确定事项/待解决问题”等固定报告栏目。
"""


FALLBACK_SYSTEM_PROMPT = """请把下面的 Telegram 群聊整理成一份正常的中文群聊总结。
不要输出 JSON，不要输出代码块，不要输出消息 ID，不要解释规则。
直接输出：
📝 群聊 AI 总结

📌 主题一
用 1-3 句话概括

📌 主题二
用 1-3 句话概括

💡 总体结论
用 1-2 句话概括整体聊天。

忽略“哈哈”“6”“好的”“收到”、纯表情、纯打招呼等无意义消息。
只总结真实出现的内容。
"""


def _clean_json_candidate(raw: str) -> str:
    """Remove common Markdown fences / surrounding prose before JSON parsing."""
    text = (raw or "").strip()
    if not text:
        return ""

    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    text = text.strip()

    # Some providers prepend a short sentence before the JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text


def _build_message_link(line: ChatLine) -> str | None:
    if line.message_link:
        return line.message_link
    if line.message_id is None:
        return None

    if line.chat_username:
        username = line.chat_username.lstrip("@").strip()
        if username:
            return f"https://t.me/{username}/{line.message_id}"

    if line.chat_id is not None and line.chat_id < 0:
        raw_chat_id = str(abs(line.chat_id))
        if raw_chat_id.startswith("100"):
            raw_chat_id = raw_chat_id[3:]
        if raw_chat_id:
            return f"https://t.me/c/{raw_chat_id}/{line.message_id}"
    return None


def _known_messages(lines: Sequence[ChatLine]) -> dict[int, str | None]:
    return {
        int(line.message_id): _build_message_link(line)
        for line in lines
        if line.message_id is not None
    }


def _extract_conclusion_from_partial_json(raw: str) -> str:
    match = re.search(r'"conclusion"\s*:\s*"((?:\\.|[^"\\])*)"', raw, flags=re.DOTALL)
    if not match:
        return ""
    try:
        return str(json.loads('"' + match.group(1) + '"')).strip()
    except json.JSONDecodeError:
        return ""


def _parse_complete_json(raw: str) -> dict | None:
    candidate = _clean_json_candidate(raw)
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _recover_complete_topics(raw: str) -> tuple[list[dict], str]:
    """Recover completed topic objects when the model was cut off mid-JSON.

    This deliberately ignores an unfinished final topic instead of showing half a
    sentence or internal JSON to users.
    """
    candidate = _clean_json_candidate(raw)
    match = re.search(r'"topics"\s*:\s*\[', candidate, flags=re.IGNORECASE)
    if not match:
        return [], ""

    decoder = JSONDecoder()
    cursor = match.end()
    recovered: list[dict] = []

    while cursor < len(candidate):
        while cursor < len(candidate) and candidate[cursor].isspace():
            cursor += 1
        if cursor >= len(candidate) or candidate[cursor] == "]":
            break
        if candidate[cursor] == ",":
            cursor += 1
            continue
        if candidate[cursor] != "{":
            break
        try:
            item, end = decoder.raw_decode(candidate, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict):
            recovered.append(item)
        cursor = end

    return recovered, _extract_conclusion_from_partial_json(candidate)


def _coerce_message_id(item: dict, known: dict[int, str | None]) -> int | None:
    candidates: list[object] = []
    if "message_id" in item:
        candidates.append(item.get("message_id"))
    ids = item.get("message_ids")
    if isinstance(ids, list):
        candidates.extend(ids[:3])
    elif ids is not None:
        candidates.append(ids)

    for value in candidates:
        try:
            message_id = int(value)
        except (TypeError, ValueError):
            continue
        if message_id in known:
            return message_id
    return None


def _normalise_title(value: object) -> str:
    title = str(value or "").strip()
    title = re.sub(r"[\r\n]+", " ", title)
    title = re.sub(r"\s{2,}", " ", title)
    return truncate_text(title, MAX_TOPIC_TITLE_LENGTH).strip()


def _normalise_description(value: object) -> str:
    description = str(value or "").strip()
    description = re.sub(r"^```(?:text|markdown)?\s*", "", description, flags=re.IGNORECASE)
    description = re.sub(r"\s*```$", "", description)
    description = re.sub(r"[\r\n]{3,}", "\n\n", description)
    description = description.strip()

    # Never allow raw JSON-in-a-description to leak to the user.
    if description.startswith("{") and '"topics"' in description:
        return ""
    return truncate_text(description, MAX_TOPIC_DESCRIPTION_LENGTH).strip()


def _parse_summary(raw: str, lines: Sequence[ChatLine]) -> ChatSummary | None:
    known = _known_messages(lines)
    data = _parse_complete_json(raw)

    if data is not None:
        topic_items = data.get("topics")
        conclusion = str(data.get("conclusion") or "").strip()
        if not isinstance(topic_items, list):
            topic_items = []
    else:
        topic_items, recovered_conclusion = _recover_complete_topics(raw)
        conclusion = recovered_conclusion

    topics: list[SummaryTopic] = []
    seen_titles: set[str] = set()

    for item in topic_items:
        if not isinstance(item, dict):
            continue
        title = _normalise_title(item.get("title"))
        description = _normalise_description(item.get("description") or item.get("summary"))
        if not title or not description:
            continue

        message_id = _coerce_message_id(item, known)
        message_link = known.get(message_id) if message_id is not None else None

        title_key = title.casefold()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        topics.append(
            SummaryTopic(
                title=title,
                description=description,
                message_id=message_id,
                message_link=message_link,
            )
        )
        if len(topics) >= MAX_TOPICS:
            break

    if not topics:
        return None

    if not conclusion:
        titles = [f"“{topic.title}”" for topic in topics[:4]]
        if len(titles) == 1:
            conclusion = f"本次聊天主要围绕{'、'.join(titles)}等话题展开。"
        elif len(titles) > 1:
            conclusion = f"本次聊天主要围绕{'、'.join(titles)}等话题展开。"

    conclusion = re.sub(r"[\r\n]{3,}", "\n\n", conclusion).strip()
    conclusion = truncate_text(conclusion, 500).strip()

    return ChatSummary(tuple(topics), conclusion)


def render_chat_summary(summary: ChatSummary) -> str:
    lines = ["📝 群聊 AI 总结", ""]

    for topic in summary.topics:
        if topic.message_link:
            title = f"[📌 {topic.title}]({topic.message_link})"
        else:
            title = f"📌 {topic.title}"
        lines.extend([title, topic.description, ""])

    if summary.conclusion:
        lines.extend(["💡 总体结论", summary.conclusion])

    return "\n".join(lines).strip()


def _render_plain_fallback_from_text(raw: str) -> str:
    """Last-resort readable fallback that cannot expose JSON internals."""
    text = (raw or "").strip()
    if not text:
        return "📝 群聊 AI 总结\n\n暂时没有生成可显示的总结内容。"

    # If the provider returned fenced JSON or JSON fragments, never forward them.
    if "\"topics\"" in text or text.lstrip().startswith("{"):
        return "📝 群聊 AI 总结\n\n暂时无法稳定解析本次总结，已保留聊天记录，稍后可重新执行 /summary。"

    text = re.sub(r"```(?:text|markdown)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()
    text = re.sub(r"(?:message_ids?|message_indexes)\s*[:=].*", "", text, flags=re.IGNORECASE)
    text = truncate_text(text, 2500).strip()
    if not text:
        return "📝 群聊 AI 总结\n\n暂时无法生成可显示的总结内容。"

    return f"📝 群聊 AI 总结\n\n📌 群聊讨论\n{text}\n\n💡 总体结论\n本次聊天总结已生成。"


async def summarize_chat_result(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> ChatSummary | str:
    valid_lines = [line for line in lines if line.text and line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条消息。"

    content_parts: list[str] = []
    for line in valid_lines:
        # ID is supplied solely for internal link mapping. The model is asked to
        # return only one source ID per topic so the JSON stays small and stable.
        source = f" [message_id={line.message_id}]" if line.message_id is not None else ""
        author = f"{line.author}: " if line.author else ""
        content_parts.append(f"{author}{line.text}{source}")

    content = truncate_text("\n".join(content_parts), 20000)

    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
            max_tokens=2800,
        )
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc

    parsed = _parse_summary(raw, valid_lines)
    if parsed is not None:
        return parsed

    # One cheap recovery pass: use normal Markdown rather than asking the model
    # for JSON again. This protects the user-facing channel from malformed JSON.
    try:
        fallback_raw = await ai.chat(
            [
                {"role": "system", "content": FALLBACK_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
            max_tokens=1800,
        )
    except AIError:
        return _render_plain_fallback_from_text(raw)

    # Prefer the readable fallback only after sanitising out any accidental
    # internal fields / code fences. It is already user-facing Markdown.
    cleaned = _render_plain_fallback_from_text(fallback_raw)
    if "\"topics\"" in cleaned:
        return _render_plain_fallback_from_text(raw)
    return ChatSummary(
        topics=(
            SummaryTopic(
                title="群聊讨论",
                description=cleaned.split("📌 群聊讨论\n", 1)[-1].split("\n\n💡 总体结论", 1)[0].strip(),
            ),
        ),
        conclusion="本次聊天总结已生成。",
    )


async def summarize_chat(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> str:
    result = await summarize_chat_result(ai, lines, model, automatic=automatic)
    if isinstance(result, str):
        return result
    return render_chat_summary(result)
