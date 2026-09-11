from app.handlers import BotState, MAX_STORED_CONTEXT_MESSAGES


class DummySettings:
    context_max_messages = 8
    default_model = "test-model"


def test_group_context_is_shared_between_users():
    state = BotState(DummySettings(), None, None, None)
    state.add_group_message(123, "甲", "今天27度")
    # A second user sees the same group context.
    class Chat: pass
    class User: pass
    class Update: pass
    u = Update(); u.effective_chat = Chat(); u.effective_chat.id = 123; u.effective_chat.type = "group"; u.effective_user = User(); u.effective_user.id = 999
    assert any("今天27度" in item["content"] for item in state.recent_context(u))


def test_bot_answer_is_available_to_later_group_turn():
    state = BotState(DummySettings(), None, None, None)
    state.add_group_assistant(123, "我刚才回答：27°C 左右。")
    assert "27°C" in state.group_contexts[123][0]["content"]


def test_group_context_keeps_author_and_assistant_turns_together():
    state = BotState(DummySettings(), None, None, None)
    state.add_group_message(456, "甲", "今天27度")
    state.add_group_message(456, "乙", "那晚上会不会降温？")
    state.add_group_assistant(456, "根据目前上下文，你们讨论的是今天的气温。")
    contents = [item["content"] for item in state.group_contexts[456]]
    assert contents[0].startswith("甲：")
    assert "27度" in contents[0]
    assert contents[1].startswith("乙：")
    assert contents[2].startswith("根据目前上下文")


def test_group_context_has_bounded_memory():
    state = BotState(DummySettings(), None, None, None)
    for i in range(MAX_STORED_CONTEXT_MESSAGES + 10):
        state.add_group_message(789, "甲", f"消息{i}")
    assert len(state.group_contexts[789]) == MAX_STORED_CONTEXT_MESSAGES
    assert "消息0" not in " ".join(x["content"] for x in state.group_contexts[789])


def test_replying_to_another_user_does_not_trigger_bot():
    from types import SimpleNamespace
    from app.handlers import should_handle_group_message

    state = BotState(DummySettings(), None, None, None)
    state.bot_username = "my_test_bot"
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_message=SimpleNamespace(
            text="这是什么意思？",
            caption=None,
            photo=None,
            reply_to_message=SimpleNamespace(
                from_user=SimpleNamespace(username="another_user")
            ),
        ),
    )
    assert should_handle_group_message(update, state) is False


def test_replying_to_bot_does_trigger_bot():
    from types import SimpleNamespace
    from app.handlers import should_handle_group_message

    state = BotState(DummySettings(), None, None, None)
    state.bot_username = "my_test_bot"
    update = SimpleNamespace(
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_message=SimpleNamespace(
            text="继续说",
            caption=None,
            photo=None,
            reply_to_message=SimpleNamespace(
                from_user=SimpleNamespace(username="my_test_bot")
            ),
        ),
    )
    assert should_handle_group_message(update, state) is True
