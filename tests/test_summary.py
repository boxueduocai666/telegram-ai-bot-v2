from datetime import datetime, timezone

from app.summary import ChatLine, render_summary


def test_render_summary_v1_style_markdown() -> None:
    lines = [
        ChatLine(
            role="user",
            author="bo xueduocai",
            text="分享了一个 GitHub 项目链接",
            chat_id=-100123456,
            message_id=10,
            chat_username="example_group",
            timestamp=datetime(2026, 9, 12, 4, 58, tzinfo=timezone.utc),
        ),
        ChatLine(
            role="user",
            author="bo xueduocai",
            text="后来烟雨皆散尽，无人撑伞一人行",
            chat_id=-100123456,
            message_id=11,
            chat_username="example_group",
            timestamp=datetime(2026, 9, 12, 4, 59, tzinfo=timezone.utc),
        ),
        ChatLine(
            role="user",
            author="bo xueduocai",
            text="我与旧事归于尽，来年依旧迎花开",
            chat_id=-100123456,
            message_id=12,
            chat_username="example_group",
            timestamp=datetime(2026, 9, 12, 4, 59, tzinfo=timezone.utc),
        ),
    ]
    data = {
        "topics": [
            {
                "title": "Telegram AI Bot 项目分享",
                "summary": "用户 bo xueduocai 分享了一个 GitHub 项目链接。",
                "message_indexes": [1],
            },
            {
                "title": "古风诗句与情感文案分享",
                "summary": "用户连续发送了多条古风与抒情文案，整体风格偏向情感抒怀。",
                "message_indexes": [2, 3],
            },
        ],
        "overall_conclusion": "该群聊近期互动较少，主要由 bo xueduocai 活跃，内容偏向项目分享和文艺表达。",
    }

    output = render_summary(data, lines)
    assert "📝 **群聊 AI 总结**" in output
    assert "💬 已分析 3 条有效消息，整理出 2 个主要话题" in output
    assert "📌" in output
    assert "💡 **总体结论**" in output
    assert "[📌 Telegram AI Bot 项目分享](https://t.me/example_group/10)" in output
