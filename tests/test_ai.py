import pytest

from app.ai import AIClient, AIError, clean_model_output


class FakeCompletions:
    async def create(self, **kwargs):
        class Msg:
            content = "hello"

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        return Response()


class FakeReasoningCompletions:
    async def create(self, **kwargs):
        class Msg:
            content = "<think>内部分析：这里不应该发送给用户。</think>\n\n最终答案：hello"

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        return Response()


class FakeClient:
    class chat:
        completions = FakeCompletions()


class FakeReasoningClient:
    class chat:
        completions = FakeReasoningCompletions()


@pytest.mark.asyncio
async def test_ai_chat():
    ai = AIClient("x", "https://example.com/v1")
    ai.client = FakeClient()
    result = await ai.chat([{"role": "user", "content": "hi"}], "model")
    assert result == "hello"


@pytest.mark.asyncio
async def test_ai_chat_removes_reasoning_block():
    ai = AIClient("x", "https://example.com/v1")
    ai.client = FakeReasoningClient()
    result = await ai.chat([{"role": "user", "content": "hi"}], "model")
    assert result == "最终答案：hello"
    assert "内部分析" not in result
    assert "<think>" not in result


def test_clean_model_output_multiple_reasoning_formats():
    text = (
        "<analysis>分析一</analysis>\n"
        "<reasoning>分析二</reasoning>\n\n"
        "# 结果\n这是给用户看的答案。"
    )
    assert clean_model_output(text) == "# 结果\n这是给用户看的答案。"
