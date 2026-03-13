"""Tests for the LLM chat completion client."""

from __future__ import annotations

import json

import httpx
import pytest
from services.agent.llm_client import LlmClient


def _make_response(body: dict, status: int = 200) -> httpx.Response:
    return httpx.Response(status_code=status, json=body)


class _FakeTransport(httpx.BaseTransport):
    """Return a canned response for every request."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.last_request: httpx.Request | None = None

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.last_request = request
        return self._response


def _client_with(
    response_body: dict, *, status: int = 200
) -> tuple[LlmClient, _FakeTransport]:
    transport = _FakeTransport(_make_response(response_body, status))
    http = httpx.Client(transport=transport, base_url="http://fake")
    client = LlmClient(base_url="http://fake", model="test-model", http_client=http)
    return client, transport


class TestChat:
    def test_plain_text_response(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!",
                    }
                }
            ]
        }
        client, transport = _client_with(body)
        result = client.chat([{"role": "user", "content": "hi"}])

        assert result["content"] == "Hello there!"
        assert result.get("tool_calls") is None

        assert transport.last_request is not None
        sent = json.loads(transport.last_request.content)
        assert sent["model"] == "test-model"
        assert sent["messages"] == [{"role": "user", "content": "hi"}]
        assert "tools" not in sent

    def test_tool_calls_response(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "search_chunks",
                                    "arguments": '{"query": "multa"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
        client, _ = _client_with(body)
        result = client.chat(
            [{"role": "user", "content": "buscar multas"}],
            tools=[{"type": "function", "function": {"name": "search_chunks"}}],
        )

        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search_chunks"

    def test_sends_tools_when_provided(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        client, transport = _client_with(body)
        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        client.chat([{"role": "user", "content": "test"}], tools=tools)

        assert transport.last_request is not None
        sent = json.loads(transport.last_request.content)
        assert sent["tools"] == tools

    def test_http_error_raises(self) -> None:
        client, _ = _client_with({"error": "bad"}, status=500)
        with pytest.raises(httpx.HTTPStatusError):
            client.chat([{"role": "user", "content": "fail"}])

    def test_respects_max_tokens_and_temperature(self) -> None:
        body = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        transport = _FakeTransport(_make_response(body))
        http = httpx.Client(transport=transport, base_url="http://fake")
        client = LlmClient(
            base_url="http://fake",
            model="m",
            max_tokens=128,
            temperature=0.5,
            http_client=http,
        )
        client.chat([{"role": "user", "content": "x"}])
        assert transport.last_request is not None
        sent = json.loads(transport.last_request.content)
        assert sent["max_tokens"] == 128
        assert sent["temperature"] == 0.5
