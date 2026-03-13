"""
Minimal HTTP client for OpenAI-compatible /v1/chat/completions.

Follows the same dataclass + httpx pattern used by
services/search/embedding_client.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class LlmClient:
    """Chat completion client with tool-calling support."""

    base_url: str
    model: str
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout_seconds: float = 300.0
    api_key: str = ""
    http_client: httpx.Client | None = field(default=None, repr=False)

    def _client(self) -> httpx.Client:
        if self.http_client is not None:
            return self.http_client
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return httpx.Client(
            base_url=self.base_url, timeout=self.timeout_seconds, headers=headers
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return the assistant message dict.

        When the model decides to call tools, the returned dict will contain
        a ``tool_calls`` key with the list of tool invocations.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = tools

        client = self._client()
        owns_client = self.http_client is None
        try:
            response = client.post("/v1/chat/completions", json=body)
            if response.status_code >= 400:
                logger.error(
                    "LLM API error %d: %s",
                    response.status_code,
                    response.text[:2000],
                )
                response.raise_for_status()

            payload = response.json()
            message: dict[str, Any] = payload["choices"][0]["message"]
            return message
        finally:
            if owns_client:
                client.close()
