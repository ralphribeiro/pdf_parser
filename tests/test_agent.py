"""Tests for the ReAct agent loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from services.agent.agent import (
    AgentResult,
    AgentRunner,
    _parse_text_tool_calls,
    _strip_tool_call_markup,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeLlm:
    """LLM stub that returns pre-programmed responses in sequence."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.messages_log: list[list[dict]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.messages_log.append(messages)
        resp = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return resp


@dataclass
class _FakeToolRegistry:
    """Tool registry stub with recorded calls."""

    results: dict[str, str] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_chunks",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def execute(
        self, tool_name: str, arguments: dict[str, Any], max_chars: int = 0
    ) -> str:
        self.calls.append((tool_name, arguments))
        return self.results.get(tool_name, "result")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRunner:
    def test_direct_answer(self) -> None:
        """LLM answers immediately without calling tools."""
        llm = _FakeLlm([{"role": "assistant", "content": "A resposta e 42."}])
        tools = _FakeToolRegistry()
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=5, context_budget=100000
        )

        result = agent.run("qual e a resposta?")

        assert isinstance(result, AgentResult)
        assert result.answer == "A resposta e 42."
        assert result.iterations == 1
        assert len(tools.calls) == 0

    def test_single_tool_call_then_answer(self) -> None:
        """LLM calls one tool, then answers."""
        llm = _FakeLlm(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": json.dumps({"query": "multa"}),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Encontrei clausulas de multa na pagina 3.",
                },
            ]
        )
        tools = _FakeToolRegistry(results={"search_chunks": "chunk result"})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=5, context_budget=100000
        )

        result = agent.run("buscar multas")

        assert result.answer == "Encontrei clausulas de multa na pagina 3."
        assert result.iterations == 2
        assert len(tools.calls) == 1
        assert tools.calls[0] == ("search_chunks", {"query": "multa"})

    def test_multi_hop(self) -> None:
        """LLM makes multiple tool calls across iterations."""
        llm = _FakeLlm(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": json.dumps({"query": "multa"}),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": json.dumps({"query": "penalidade"}),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Cruzando dados: multa e penalidade na pag 5.",
                },
            ]
        )
        tools = _FakeToolRegistry(results={"search_chunks": "data"})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=5, context_budget=100000
        )

        result = agent.run("multas e penalidades")

        assert result.iterations == 3
        assert len(tools.calls) == 2

    def test_max_iterations_cap(self) -> None:
        """Agent stops after max_iterations even if LLM keeps calling tools."""
        tool_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {
                        "name": "search_chunks",
                        "arguments": json.dumps({"query": "loop"}),
                    },
                }
            ],
        }
        final = {"role": "assistant", "content": "Forced answer."}
        llm = _FakeLlm([tool_response, tool_response, tool_response, final])
        tools = _FakeToolRegistry(results={"search_chunks": "x"})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=3, context_budget=100000
        )

        result = agent.run("loop forever")

        assert result.iterations <= 4
        assert result.answer is not None

    def test_budget_exhaustion(self) -> None:
        """Agent forces answer when context budget is nearly exhausted."""
        llm = _FakeLlm(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": json.dumps({"query": "x"}),
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "Budget answer."},
            ]
        )
        tools = _FakeToolRegistry(results={"search_chunks": "a" * 500})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=10, context_budget=200
        )

        result = agent.run("test budget")
        assert result.answer is not None

    def test_text_tool_calls_are_executed(self) -> None:
        """LLM outputs <tool_call> as text — agent parses and executes them."""
        text_with_tool = (
            "Vou buscar.\n\n"
            "<tool_call>\n"
            "<function=search_chunks>\n"
            "<parameter=query>cessao credito</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        llm = _FakeLlm(
            [
                {"role": "assistant", "content": text_with_tool},
                {"role": "assistant", "content": "Encontrei dados sobre cessao."},
            ]
        )
        tools = _FakeToolRegistry(results={"search_chunks": "result"})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=5, context_budget=100000
        )

        result = agent.run("cessao de credito")

        assert len(tools.calls) == 1
        assert tools.calls[0][0] == "search_chunks"
        assert tools.calls[0][1] == {"query": "cessao credito"}
        assert "<tool_call>" not in result.answer

    def test_text_tool_calls_stripped_on_last_iteration(self) -> None:
        """When at max_iterations, <tool_call> markup is stripped from the answer."""
        text_with_tool = (
            "Encontrei dois docs.\n\n"
            "<tool_call>\n"
            "<function=search_chunks>\n"
            "<parameter=query>multa</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        llm = _FakeLlm([{"role": "assistant", "content": text_with_tool}])
        tools = _FakeToolRegistry()
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=1, context_budget=100000
        )

        result = agent.run("multa")

        assert "<tool_call>" not in result.answer
        assert "Encontrei dois docs." in result.answer

    def test_sources_extracted(self) -> None:
        """Agent extracts source info from tool call history."""
        chunk_result = json.dumps(
            {
                "chunk_id": "doc1:3:b1",
                "document_id": "doc1",
                "filename": "contrato.pdf",
                "page": 3,
                "similarity": 0.95,
                "text": "Clausula de multa",
            }
        )
        llm = _FakeLlm(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "search_chunks",
                                "arguments": json.dumps({"query": "multa"}),
                            },
                        }
                    ],
                },
                {
                    "role": "assistant",
                    "content": "Multa de 10% na pagina 3.",
                },
            ]
        )
        tools = _FakeToolRegistry(results={"search_chunks": chunk_result})
        agent = AgentRunner(
            llm=llm, tool_registry=tools, max_iterations=5, context_budget=100000
        )

        result = agent.run("multa")

        assert len(result.sources) >= 1
        assert result.sources[0]["document_id"] == "doc1"
        assert result.sources[0]["filename"] == "contrato.pdf"
        assert result.sources[0]["page"] == 3


class TestParseTextToolCalls:
    def test_single_tool_call(self) -> None:
        text = (
            "<tool_call>\n"
            "<function=search_chunks>\n"
            "<parameter=query>multa</parameter>\n"
            "<parameter=n_results>5</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0] == ("search_chunks", {"query": "multa", "n_results": "5"})

    def test_multiple_tool_calls(self) -> None:
        text = (
            "<tool_call>\n<function=get_document>\n"
            "<parameter=document_id>abc</parameter>\n"
            "</function>\n</tool_call>\n"
            "<tool_call>\n<function=search_chunks>\n"
            "<parameter=query>test</parameter>\n"
            "</function>\n</tool_call>"
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0][0] == "get_document"
        assert calls[1][0] == "search_chunks"

    def test_no_tool_calls(self) -> None:
        assert not _parse_text_tool_calls("just plain text")

    def test_multiline_parameter_value(self) -> None:
        text = (
            "<tool_call>\n"
            "<function=search_document_text>\n"
            "<parameter=document_id>\nabc123\n</parameter>\n"
            "<parameter=keyword>\ncessao credito\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = _parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0][1]["document_id"] == "abc123"
        assert calls[0][1]["keyword"] == "cessao credito"


class TestStripToolCallMarkup:
    def test_strips_tool_calls(self) -> None:
        text = "Texto valido.\n\n<tool_call>\n<function=x>\n</function>\n</tool_call>"
        assert _strip_tool_call_markup(text) == "Texto valido."

    def test_preserves_clean_text(self) -> None:
        text = "Nenhum tool call aqui."
        assert _strip_tool_call_markup(text) == "Nenhum tool call aqui."

    def test_strips_multiple(self) -> None:
        text = (
            "A\n<tool_call>\nstuff\n</tool_call>\nB\n<tool_call>\nmore\n</tool_call>\nC"
        )
        result = _strip_tool_call_markup(text)
        assert "<tool_call>" not in result
        assert "A" in result
        assert "B" in result
        assert "C" in result
