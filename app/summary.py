from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .ai import AIClient, clean_model_output


@dataclass(frozen=True)
class ChatLine:
    role: str
    text: str
    author: str = "用户"
    message_id: int | None = None
    chat_id: int | None = None
    chat_username: str | None = None
    timestamp: datetime | None = None


_IGNORED = {
    "你好", "嗨", "哈喽", "hello", "hi", "早", "早上好", "晚上好", "晚安",
    "哈哈", "哈哈哈", "哈哈哈哈", "嗯", "哦", "噢", "啊", "好的", "好", "收到",
    "ok", "OK", "666", "6",
}


def _meaningful(line: ChatLine) -> bool:
    text = (line.text or "").strip()
    if not text or text in _IGNORED or len(text) <= 1:
        return False
    if re.fullmatch(r"[\W_]+", text, flags=re.UNICODE):
        return False
    return True


def _message_link(line: ChatLine) -> str | None:
    if not line.message_id or not line.chat_id:
        return None
    if line.chat_username:
        return f"https://t.me/{line.chat_username.lstrip('@')}/{line.message_id}"
    chat_id = str(line.chat_id)
    if chat_id.startswith("-100"):
        return f"https://t.me/c/{chat_id[4:]}/{line.message_id}"
    return None


def _format_transcript(lines: Iterable[ChatLine]) -> str:
    output: list[str] = []
    for index, line in enumerate((line for line in lines if _meaningful(line)), start=1):
        timestamp = line.timestamp.strftime("%H:%M") if line.timestamp else ""
        prefix = f"{timestamp} " if timestamp else ""
        link = _message_link(line)
        source = f" [原消息]({link})" if link else ""
        output.append(f"{index}. {prefix}{line.author}：{line.text}{source}")
    return "\n".join(output)


async def summarize_chat(
    ai: AIClient,
    lines: list[ChatLine],
    model: str,
    *,
    automatic: bool = False,
) -> str:
    useful = [line for line in lines if _meaningful(line)]
    if len(useful) < 3:
        raise ValueError("not enough meaningful messages for summary")

    transcript = _format_transcript(useful[-120:])
    mode = "每日自动群聊总结" if automatic else "群聊总结"
    system_prompt = (
        "你是一个 Telegram 群聊总结助手。"
        "只根据提供的群聊记录总结，不补造不存在的信息。"
        "忽略纯问候、哈哈、收到、OK、单个表情、纯标点等无信息消息。"
        "输出简洁、自然、像真人写的中文 Markdown。"
        "不要输出 <think>、<analysis>、<reasoning> 等内部推理标记。"
    )
    user_prompt = f"""请生成一份{mode}。

严格使用下面的结构：
📝 群聊 AI 总结

📌 话题一：<话题名称>
- <关键讨论、事实或观点>
- <重要补充>

📌 话题二：<话题名称>
- <关键讨论、事实或观点>

💡 总体结论
<用 1～3 句话概括今天最重要的内容、共识或未解决问题。>

要求：
1. 只保留真正有信息量的话题；没有第二个话题时不要硬凑。
2. 不要虚构人物观点、结论或行动。
3. 能确定发言人时可写出名字。
4. 需要引用原消息时，尽量保留原记录中的 [原消息](...) 链接，不要把链接改成裸 URL。
5. 不要在末尾额外添加“参考资料”或“来源”列表。

群聊记录：
{transcript}
"""

    result = await ai.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=0.2,
        max_tokens=2500,
    )
    return clean_model_output(result)
