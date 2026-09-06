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


SUMMARY_SYSTEM_PROMPT = """你是 Telegram 群聊日报总结助手。

请根据提供的群聊消息，生成一份自然、简洁、好读的群聊总结。
重点是还原聊天中真正讨论过的话题，不要为了凑栏目编造“已确定事项”或“待解决问题”。

请严格输出 JSON，不要输出 Markdown、代码块或任何额外文字，格式必须是：
{
  "topics": [
    {
      "title": "简短的主题标题",
      "description": "用 1-2 句话自然概括这个话题实际聊了什么",
      "message_ids": [123, 456]
    }
  ],
  "conclusion": "用 1-2 句话概括整体聊天情况"
}

规则：
1. topics 只保留真正值得总结的讨论主题，按重要程度排序。
2. 每个主题必须引用至少一个提供的 message_id；只能使用输入中真实存在的 message_id，绝不能自己编造。
3. description 只能根据聊天内容总结，不得补充不存在的事实。
4. 如果只是简单提及某个东西，也可以保留，但要明确说“提及/话题引入”，不要假装已经展开讨论。
5. conclusion 简洁自然，不要重复所有主题。
6. 如果聊天内容很少、讨论很浅，就保持简短，不要强行扩写。
7. 不要输出“最近讨论主题 / 主要观点 / 已确定事项 / 待解决问题”这类固定报告栏目。"""


def _parse_summary(raw: str, lines: Sequence[ChatLine]) -> ChatSummary | None:
    """Parse the model's JSON and validate message IDs against the supplied transcript."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    known = {
        line.message_id: line.message_link
        for line in lines
        if line.message_id is not None
    }
    topics_raw = data.get("topics")
    if not isinstance(topics_raw, list):
        return None

    topics: list[SummaryTopic] = []
    for item in topics_raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        description = str(item.get("description", "")).strip()
        if not title or not description:
            continue

        ids = item.get("message_ids", [])
        if not isinstance(ids, list):
            ids = []
        valid_ids = []
        for value in ids:
            try:
                message_id = int(value)
            except (TypeError, ValueError):
                continue
            if message_id in known:
                valid_ids.append(message_id)

        source_id = valid_ids[0] if valid_ids else None
        topics.append(
            SummaryTopic(
                title=title,
                description=description,
                message_id=source_id,
                message_link=known.get(source_id) if source_id is not None else None,
            )
        )

    conclusion = str(data.get("conclusion", "")).strip()
    if not topics and not conclusion:
        return None
    return ChatSummary(tuple(topics), conclusion)


def render_chat_summary(summary: ChatSummary) -> str:
    """Render the user-facing V1-style Telegram summary.

    Topic titles are Markdown links when a reliable Telegram message link exists.
    There is intentionally no ugly URL/link section at the bottom.
    """
    lines = ["📝 群聊 AI 总结", ""]
    for topic in summary.topics:
        title = f"[📌 {topic.title}]({topic.message_link})" if topic.message_link else f"📌 {topic.title}"
        lines.extend([title, topic.description, ""])
    if summary.conclusion:
        lines.extend(["💡 总体结论", summary.conclusion])

    return "\n".join(lines).strip()


async def summarize_chat(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
) -> str:
    """Return a V1-style summary. Kept as a compatibility API for existing callers/tests."""
    result = await summarize_chat_result(ai, lines, model)
    if isinstance(result, str):
        return result
    return render_chat_summary(result)


async def summarize_chat_result(
    ai: AIClient,
    lines: Sequence[ChatLine],
    model: str,
) -> ChatSummary | str:
    valid_lines = [line for line in lines if line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条消息。"

    content_parts = []
    for line in valid_lines:
        source = f" [message_id={line.message_id}]" if line.message_id is not None else ""
        content_parts.append(f"{line.author + ': ' if line.author else ''}{line.text}{source}")
    content = truncate_text("\n".join(content_parts), 20000)

    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
        )
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc

    parsed = _parse_summary(raw, valid_lines)
    if parsed is not None:
        return parsed

    # Graceful fallback: preserve the old V1-like presentation if a provider
    # ignores the JSON instruction. No guessed links are created in this case.
    return ChatSummary(
        topics=(
            SummaryTopic(
                title="群聊讨论",
                description=truncate_text(raw.strip(), 1000),
            ),
        ),
        conclusion="以上内容为 AI 根据当前聊天记录生成的总结。",
    )
