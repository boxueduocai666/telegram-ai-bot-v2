from unittest.mock import AsyncMock

import pytest

from app.summary import ChatLine, summarize_chat


@pytest.mark.asyncio
async def test_summary_allows_one_author_with_three_messages():
    ai = AsyncMock()
    ai.chat.return_value = "总结"
    lines = [
        ChatLine("user", "第一句", "小明"),
        ChatLine("user", "第二句", "小明"),
        ChatLine("user", "第三句", "小明"),
    ]

    result = await summarize_chat(ai, lines, "model-a")

    assert result == "总结"
    ai.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_summary_rejects_fewer_than_three_messages():
    ai = AsyncMock()
    lines = [
        ChatLine("user", "第一句", "小明"),
        ChatLine("user", "第二句", "小明"),
    ]

    result = await summarize_chat(ai, lines, "model-a")

    assert "至少需要 3 条消息" in result
    ai.chat.assert_not_awaited()
