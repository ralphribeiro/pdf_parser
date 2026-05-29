"""
ReAct agent loop with context-budget tracking.

Iteratively calls the LLM, executes tool calls, and appends results
until the LLM produces a final text answer or limits are reached.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from services.agent.prompts import FORCE_ANSWER_ADDENDUM, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4

_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.+?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=(\w+)>\s*(.+?)\s*</function>", re.DOTALL)
_PARAMETER_RE = re.compile(r"<parameter=(\w+)>\s*(.+?)\s*</parameter>", re.DOTALL)


def _parse_text_tool_calls(content: str) -> list[tuple[str, dict[str, str]]]:
    """Extract tool calls from Qwen-style ``<tool_call>`` text markup.

    Returns a list of ``(function_name, {param: value})`` tuples.
    """
    results: list[tuple[str, dict[str, str]]] = []
    for block_match in _TOOL_CALL_BLOCK_RE.finditer(content):
        block = block_match.group(1)
        fn_match = _FUNCTION_RE.search(block)
        if not fn_match:
            continue
        fn_name = fn_match.group(1)
        fn_body = fn_match.group(2)
        params: dict[str, str] = {}
        for param_match in _PARAMETER_RE.finditer(fn_body):
            params[param_match.group(1)] = param_match.group(2).strip()
        results.append((fn_name, params))
    return results


def _strip_tool_call_markup(content: str) -> str:
    """Remove ``<tool_call>…</tool_call>`` blocks from text."""
    cleaned = _TOOL_CALL_BLOCK_RE.sub("", content)
    return re.sub(r"\n{3,}", "\n", cleaned).strip()


@dataclass
class AgentResult:
    answer: str
    sources: list[dict[str, Any]]
    iterations: int


def _estimate_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(content)
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total += len(fn.get("name", "")) + len(fn.get("arguments", ""))
    return total


def _extract_sources(tool_results: list[str]) -> list[dict[str, Any]]:
    """Parse structured source entries from tool result strings.

    Handles results from both ``search_chunks`` (has ``chunk_id``) and
    ``search_document_text`` (has ``block_id`` mapped to ``chunk_id``).
    """
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for text in tool_results:
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            doc_id = obj.get("document_id", "")
            chunk_id = obj.get("chunk_id") or obj.get("block_id") or ""
            page = obj.get("page", 0)
            has_content = chunk_id or page or obj.get("text")
            if not doc_id or not has_content:
                continue
            dedup_key = f"{doc_id}:{chunk_id}:{page}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            sources.append(
                {
                    "document_id": doc_id,
                    "filename": obj.get("filename", ""),
                    "chunk_id": chunk_id,
                    "page": page,
                    "text": obj.get("text", ""),
                }
            )
    return sources


def _run_messages_and_registry(
    query: str,
    document_id: str | None,
    tool_registry: Any,
) -> tuple[list[dict[str, Any]], Any, list[dict[str, Any]]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    active_registry = (
        tool_registry.scoped_to_document(document_id) if document_id else tool_registry
    )
    if document_id:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use somente o documento "
                    f"{document_id}; as ferramentas ja estao restritas a ele."
                ),
            }
        )
    messages.append({"role": "user", "content": query})
    return messages, active_registry, active_registry.tool_schemas()


def _budget_force_answer(
    llm: Any,
    messages: list[dict[str, Any]],
    tool_results: list[str],
    iteration: int,
) -> AgentResult:
    messages.append({"role": "user", "content": FORCE_ANSWER_ADDENDUM})
    response = llm.chat(messages, tools=None)
    return AgentResult(
        answer=_strip_tool_call_markup(response.get("content") or ""),
        sources=_extract_sources(tool_results),
        iterations=iteration,
    )


def _run_text_tool_calls(
    messages: list[dict[str, Any]],
    content: str,
    text_calls: list[tuple[str, dict[str, str]]],
    active_registry: Any,
    tool_results: list[str],
    budget_remaining: int,
) -> None:
    messages.append({"role": "assistant", "content": content})
    per_call = max(budget_remaining // max(len(text_calls), 1), 1000)
    for fn_name, fn_args in text_calls:
        result_text = active_registry.execute(fn_name, fn_args, max_chars=per_call)
        tool_results.append(result_text)
        messages.append(
            {
                "role": "user",
                "content": f"[Tool result: {fn_name}]\n{result_text}",
            }
        )


def _run_structured_tool_calls(
    messages: list[dict[str, Any]],
    response: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    active_registry: Any,
    tool_results: list[str],
    budget_remaining: int,
) -> None:
    messages.append(response)
    per_call = max(budget_remaining // max(len(tool_calls), 1), 1000)
    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        raw_args = fn.get("arguments", {})
        if isinstance(raw_args, dict):
            args = raw_args
        else:
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {}
        result_text = active_registry.execute(name, args, max_chars=per_call)
        tool_results.append(result_text)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result_text,
            }
        )


def _final_force_answer(
    llm: Any,
    messages: list[dict[str, Any]],
    tool_results: list[str],
    iteration: int,
) -> AgentResult:
    messages.append({"role": "user", "content": FORCE_ANSWER_ADDENDUM})
    try:
        response = llm.chat(messages, tools=None)
        answer = _strip_tool_call_markup(response.get("content") or "")
    except Exception:
        logger.exception("Force-answer LLM call failed, returning collected data")
        answer = ""
    return AgentResult(
        answer=answer,
        sources=_extract_sources(tool_results),
        iterations=iteration + 1,
    )


@dataclass
class AgentRunner:
    llm: Any
    tool_registry: Any
    max_iterations: int = 8
    context_budget: int = 400_000

    _tool_results: list[str] = field(default_factory=list, init=False, repr=False)

    def run(self, query: str, document_id: str | None = None) -> AgentResult:
        self._tool_results = []
        messages, active_registry, tools = _run_messages_and_registry(
            query, document_id, self.tool_registry
        )

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            budget_used = _estimate_chars(messages)
            budget_remaining = self.context_budget - budget_used

            if budget_remaining < self.context_budget * 0.1:
                logger.info(
                    "Budget nearly exhausted (%d chars), forcing answer", budget_used
                )
                return _budget_force_answer(
                    self.llm, messages, self._tool_results, iteration
                )

            is_last = iteration >= self.max_iterations
            response = self.llm.chat(
                messages,
                tools=None if is_last else tools,
            )

            tool_calls = response.get("tool_calls")
            content = response.get("content") or ""

            logger.info(
                "Iter %d/%d — tool_calls=%s, content_len=%d, content_preview=%.200s",
                iteration,
                self.max_iterations,
                bool(tool_calls),
                len(content),
                content,
            )

            text_calls = _parse_text_tool_calls(content) if content else []

            if not tool_calls and not text_calls:
                return AgentResult(
                    answer=_strip_tool_call_markup(content),
                    sources=_extract_sources(self._tool_results),
                    iterations=iteration,
                )

            if text_calls and not tool_calls:
                logger.info("Parsed %d tool call(s) from text content", len(text_calls))
                _run_text_tool_calls(
                    messages,
                    content,
                    text_calls,
                    active_registry,
                    self._tool_results,
                    budget_remaining,
                )
                if is_last:
                    break
            else:
                _run_structured_tool_calls(
                    messages,
                    response,
                    tool_calls,
                    active_registry,
                    self._tool_results,
                    budget_remaining,
                )

            n_calls = len(tool_calls) if tool_calls else len(text_calls)
            logger.info(
                "Iteration %d: %d tool calls, budget %d/%d chars",
                iteration,
                n_calls,
                _estimate_chars(messages),
                self.context_budget,
            )

        return _final_force_answer(self.llm, messages, self._tool_results, iteration)
