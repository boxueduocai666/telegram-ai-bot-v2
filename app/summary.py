from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .ai import AIClient, AIError
from .utils import truncate_text


@dataclass(frozen=True)
class ChatLine:
    role: str
    text: str
    author: str = ""


MIN_SUMMARY_MESSAGES = 3


SUMMARY_SYSTEM_PROMPT = """你是群聊总结助手。请基于提供的聊天内容进行客观总结。
输出 Markdown，包含：
1. 最近讨论主题
2. 主要观点
3. 已确定事项
4. 待解决问题（没有则写“暂无”）
不要补充聊天记录中不存在的事实。"""


async def summarize_chat(ai: AIClient, lines: Sequence[ChatLine], model: str) -> str:
    valid_lines = [line for line in lines if line.text.strip()]
    if len(valid_lines) < MIN_SUMMARY_MESSAGES:
        return f"暂时没有足够的聊天内容可以总结，至少需要 {MIN_SUMMARY_MESSAGES} 条消息。"

    content = "\n".join(
        f"{line.author + ': ' if line.author else ''}{line.text}" for line in valid_lines
    )
    content = truncate_text(content, 20000)
    try:
        return await ai.chat(
            [
                {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.2,
        )
    except AIError as exc:
        raise RuntimeError(str(exc)) from exc
