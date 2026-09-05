import pytest

from app.ai import AIClient, AIError


class FakeCompletions:
    async def create(self, **kwargs):
        class Msg:
            content = "hello"

        class Choice:
            message = Msg()

        class Response:
            choices = [Choice()]

        return Response()


class FakeClient:
    class chat:
        completions = FakeCompletions()


@pytest.mark.asyncio
async def test_ai_chat():
    ai = AIClient("x", "https://example.com/v1")
    ai.client = FakeClient()
    result = await ai.chat([{"role": "user", "content": "hi"}], "model")
    assert result == "hello"
