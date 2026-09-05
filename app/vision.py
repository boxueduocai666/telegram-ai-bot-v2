from __future__ import annotations

import base64
import logging
from io import BytesIO

from telegram import Message

from .ai import AIClient, AIError
from .utils import truncate_text

logger = logging.getLogger(__name__)


class VisionError(RuntimeError):
    pass


async def download_best_photo(message: Message) -> bytes:
    if not message.photo:
        raise VisionError("No photo found in message")
    try:
        file = await message.get_bot().get_file(message.photo[-1].file_id)
        buffer = BytesIO()
        await file.download_to_memory(buffer)
        return buffer.getvalue()
    except Exception as exc:
        logger.exception("Telegram image download failed")
        raise VisionError("图片下载失败") from exc


async def image_message_to_data_url(message: Message) -> str:
    raw = await download_best_photo(message)
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


async def analyze_telegram_image(
    ai: AIClient,
    message: Message,
    question: str,
    model: str,
    *,
    system_prompt: str = "你是一个可靠的图片理解助手。仅依据图片内容回答，不要编造看不到的信息。",
) -> str:
    data_url = await image_message_to_data_url(message)
    try:
        return await ai.analyze_image(
            model=model,
            image_data_url=data_url,
            question=truncate_text(question, 6000),
            system_prompt=system_prompt,
        )
    except AIError as exc:
        raise VisionError(str(exc)) from exc
