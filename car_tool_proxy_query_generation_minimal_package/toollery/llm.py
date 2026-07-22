from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from .schemas import ToolCall, ToolSpec
from .text import term_overlap, tokenize, top_terms


_USAGE_LOG_LOCK = threading.Lock()


class Teacher(ABC):
    @abstractmethod
    def generate_queries(self, tool: ToolSpec, count: int) -> list[str]:
        raise NotImplementedError


class Verifier(ABC):
    @abstractmethod
    def select_tool(self, query: str, tools: list[ToolSpec]) -> str | None:
        raise NotImplementedError


class FinalSelector(ABC):
    @abstractmethod
    def choose_tool(self, query: str, tools: list[ToolSpec]) -> ToolCall:
        raise NotImplementedError


class HeuristicTeacher(Teacher):
    """Deterministic fallback for the paper's teacher LLM step."""

    def generate_queries(self, tool: ToolSpec, count: int) -> list[str]:
        terms = top_terms([tool.name, tool.description, json.dumps(tool.parameters)], limit=6)
        name = tool.name.replace("_", " ")
        primary = " ".join(terms[:3]) or name
        templates = [
            f"I want to {tool.description.rstrip('.')}.",
            f"Help me with {name}.",
            f"Find a tool for {primary}.",
            f"Can you handle {primary} using the right API?",
            f"I need {name} with the required parameters.",
            f"Run something that can {tool.description.rstrip('.')}.",
            f"Which function should I call for {primary}?",
            f"Use the service related to {primary}.",
        ]
        return templates[:count]


class HeuristicVerifier(Verifier):
    """Round-trip verifier based on query-to-tool textual similarity."""

    def select_tool(self, query: str, tools: list[ToolSpec]) -> str | None:
        scores = [
            (tool.name, term_overlap(query, f"{tool.name} {tool.description} {tool.parameters}"))
            for tool in tools
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        if not scores or scores[0][1] <= 0.0:
            return None
        return scores[0][0]


class HeuristicFinalSelector(FinalSelector):
    def choose_tool(self, query: str, tools: list[ToolSpec]) -> ToolCall:
        verifier = HeuristicVerifier()
        selected = verifier.select_tool(query, tools)
        if selected is None and tools:
            selected = tools[0].name
        tool = next((item for item in tools if item.name == selected), None)
        return ToolCall(
            tool_name=selected or "",
            arguments=_extract_arguments(query, tool.parameters if tool else {}),
            confidence=term_overlap(query, f"{tool.name} {tool.description}") if tool else 0.0,
        )


class OpenAICompatibleLLM(Teacher, Verifier, FinalSelector):
    """Optional adapter for OpenAI-compatible chat-completions endpoints.

    Set OPENAI_API_KEY and optionally OPENAI_BASE_URL / OPENAI_MODEL. The core
    package does not require this adapter; it is here to mirror the article's
    teacher, verifier, and final inference LLM roles.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: int = 60,
    ) -> None:
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.extra_body = extra_body if extra_body is not None else _json_env("LLM_EXTRA_BODY")
        self.timeout = int(os.getenv("LLM_TIMEOUT", timeout))

    def generate_queries(self, tool: ToolSpec, count: int) -> list[str]:
        prompt = (
            "Generate diverse natural-language user intents for this tool. "
            "Include paraphrases, implicit constraints, and ambiguous-but-valid requests. "
            f"Return JSON list of {count} strings.\nTool: {json.dumps(tool.to_dict(), ensure_ascii=False)}"
        )
        text = self._chat(prompt)
        return _json_list(text)[:count]

    def select_tool(self, query: str, tools: list[ToolSpec]) -> str | None:
        names = [tool.name for tool in tools]
        prompt = (
            "Select exactly one tool for the user query. Return only the tool name, or NONE.\n"
            f"Query: {query}\nTools: {json.dumps([t.to_dict() for t in tools], ensure_ascii=False)}"
        )
        answer = self._chat(prompt).strip()
        return answer if answer in names else None

    def choose_tool(self, query: str, tools: list[ToolSpec]) -> ToolCall:
        prompt = (
            "Choose the best tool and infer arguments. Return JSON with keys "
            "tool_name, arguments, confidence.\n"
            f"Query: {query}\nTools: {json.dumps([t.to_dict() for t in tools], ensure_ascii=False)}"
        )
        text = self._chat(prompt)
        try:
            data = json.loads(_extract_json_object(text))
            return ToolCall(
                tool_name=str(data.get("tool_name", "")),
                arguments=dict(data.get("arguments", {})),
                confidence=float(data.get("confidence", 0.0)),
            )
        except Exception:
            return HeuristicFinalSelector().choose_tool(query, tools)

    def _chat(self, prompt: str, stage: str | None = None, metadata: dict[str, Any] | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY or OPENAI_API_KEY is required for OpenAICompatibleLLM")
        if self.api_key.startswith("PASTE_") or self.api_key.endswith("_HERE"):
            raise RuntimeError("LLM api_key still looks like a placeholder; put your real provider API key in config.")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(self.extra_body)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise RuntimeError(
                f"LLM request failed with HTTP {exc.code} {exc.reason}. "
                f"model={self.model!r} base_url={self.base_url!r}. Response: {detail}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0
        content = payload["choices"][0]["message"]["content"]
        self._log_usage(
            stage=stage,
            metadata=metadata,
            usage=payload.get("usage"),
            latency_ms=latency_ms,
            response_chars=len(content),
            prompt_chars=len(prompt),
        )
        return content

    def _log_usage(
        self,
        stage: str | None,
        metadata: dict[str, Any] | None,
        usage: Any,
        latency_ms: float,
        response_chars: int,
        prompt_chars: int,
    ) -> None:
        log_path = os.getenv("LLM_USAGE_LOG")
        if not log_path:
            return
        row: dict[str, Any] = {
            "stage": stage or "chat",
            "model": self.model,
            "base_url": self.base_url,
            "latency_ms": latency_ms,
            "prompt_chars": prompt_chars,
            "response_chars": response_chars,
        }
        if metadata:
            row.update(metadata)
        if isinstance(usage, dict):
            row.update(
                {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            )
        with _USAGE_LOG_LOCK:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _json_env(name: str) -> dict[str, Any]:
    value = os.getenv(name)
    if not value:
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be valid JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{name} must decode to a JSON object")
    return data


def _extract_arguments(query: str, schema: dict[str, Any]) -> dict[str, Any]:
    args: dict[str, Any] = {}
    properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
    tokens = tokenize(query)
    numbers = re.findall(r"-?\d+(?:\.\d+)?", query)
    currency_codes = re.findall(r"\b[A-Z]{3}\b", query.upper())
    for key, spec in properties.items():
        if not tokens:
            continue
        if isinstance(spec, dict) and spec.get("type") in {"integer", "number"} and numbers:
            args[key] = float(numbers[0]) if "." in numbers[0] else int(numbers[0])
            continue
        if "currency" in key and currency_codes:
            if key.startswith("from"):
                args[key] = currency_codes[0]
            elif key.startswith("to") and len(currency_codes) > 1:
                args[key] = currency_codes[1]
            else:
                args[key] = currency_codes[0]
            continue
        key_terms = set(tokenize(key))
        if key_terms & set(tokens):
            args[key] = " ".join(tokens[-3:])
    return args


def _json_list(text: str) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = json.loads(_extract_json_array(text))
    return [str(item) for item in data if isinstance(item, str)]


def _extract_json_array(text: str) -> str:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found")
    return text[start : end + 1]


def _extract_json_object(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return text[start : end + 1]
