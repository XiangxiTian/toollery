from __future__ import annotations

from collections import defaultdict

from .embeddings import HashingTfidfEmbedder, cosine
from .schemas import ManualEntry, RetrievalHit, ToolCandidate, ToolSpec


class ProxyQueryIndex:
    """Online Toollery stage: query-to-query retrieval and tool aggregation."""

    def __init__(
        self,
        tools: list[ToolSpec],
        manual: list[ManualEntry],
        embedder: HashingTfidfEmbedder | None = None,
    ) -> None:
        if not manual:
            raise ValueError("manual must contain at least one proxy query")
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.manual = manual
        self.embedder = embedder or HashingTfidfEmbedder()
        self.embedder.fit([entry.query for entry in manual])
        self._vectors = [self.embedder.encode(entry.query) for entry in manual]

    def search(self, query: str, proxy_top_k: int = 20) -> list[RetrievalHit]:
        query_vector = self.embedder.encode(query)
        hits = [
            RetrievalHit(entry.query, entry.tool_name, cosine(query_vector, vector))
            for entry, vector in zip(self.manual, self._vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return [hit for hit in hits[:proxy_top_k] if hit.score > 0.0]

    def retrieve_tools(
        self,
        query: str,
        tool_top_k: int = 5,
        proxy_top_k: int = 20,
    ) -> list[ToolCandidate]:
        hits = self.search(query, proxy_top_k=proxy_top_k)
        grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
        for hit in hits:
            grouped[hit.tool_name].append(hit)

        candidates: list[ToolCandidate] = []
        for tool_name, tool_hits in grouped.items():
            tool = self.tools_by_name.get(tool_name)
            if tool is None:
                continue
            ranked = sorted(tool_hits, key=lambda hit: hit.score, reverse=True)
            score = sum(hit.score for hit in ranked[:3]) / min(len(ranked), 3)
            score += 0.05 * min(len(ranked), 5)
            candidates.append(ToolCandidate(tool=tool, score=score, supporting_queries=ranked[:3]))

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:tool_top_k]
