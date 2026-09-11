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


def test_render_summary_includes_time_range_count_and_clickable_topic():
    from datetime import datetime, timezone
    from app.summary import render_summary

    lines = [
        ChatLine("user", "Kelvin被封了", "Kelvin", 100, -100123, "demo_group", datetime(2026, 9, 3, 23, 9, tzinfo=timezone.utc)),
        ChatLine("user", "又被封了？", "Yaya", 101, -100123, "demo_group", datetime(2026, 9, 4, 23, 19, tzinfo=timezone.utc)),
        ChatLine("user", "是我大号", "Kelvin", 102, -100123, "demo_group", datetime(2026, 9, 4, 23, 20, tzinfo=timezone.utc)),
    ]
    result = render_summary(
        {
            "topics": [
                {"title": "Kelvin大号被封", "summary": "大家围绕账号被封展开讨论。", "message_indexes": [1, 2]},
            ]
        },
        lines,
        automatic=True,
    )
    assert "09-03 23:09 — 09-04 23:20" in result
    assert "已分析 3 条有效消息" in result
    assert "[Kelvin大号被封](https://t.me/demo_group/100)" in result
