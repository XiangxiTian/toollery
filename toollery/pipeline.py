from __future__ import annotations

from .llm import FinalSelector, HeuristicFinalSelector
from .retrieval import ProxyQueryIndex
from .schemas import ManualEntry, ToolCall, ToolCandidate, ToolSpec


class ToolleryAgent:
    """End-to-end online flow: retrieve compact tools, then select/call one."""

    def __init__(
        self,
        tools: list[ToolSpec],
        manual: list[ManualEntry],
        selector: FinalSelector | None = None,
        tool_top_k: int = 5,
        proxy_top_k: int = 20,
    ) -> None:
        self.index = ProxyQueryIndex(tools, manual)
        self.selector = selector or HeuristicFinalSelector()
        self.tool_top_k = tool_top_k
        self.proxy_top_k = proxy_top_k

    def candidates(self, query: str) -> list[ToolCandidate]:
        return self.index.retrieve_tools(
            query,
            tool_top_k=self.tool_top_k,
            proxy_top_k=self.proxy_top_k,
        )

    def run(self, query: str) -> tuple[ToolCall, list[ToolCandidate]]:
        candidates = self.candidates(query)
        call = self.selector.choose_tool(query, [candidate.tool for candidate in candidates])
        return call, candidates
