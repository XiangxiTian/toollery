from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .embeddings import HashingTfidfEmbedder, LocalHFEmbedder, OpenAICompatibleEmbedder, cosine, embedder_signature, text_digest
from .schemas import ManualEntry, RetrievalHit, ToolCandidate, ToolSpec


class ProxyQueryIndex:
    """Online Toollery stage: query-to-query retrieval and tool aggregation."""

    def __init__(
        self,
        tools: list[ToolSpec],
        manual: list[ManualEntry],
        embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder | None = None,
        embedding_cache_path: str | Path | None = None,
        force_rebuild_embeddings: bool = False,
        embedding_progress_callback: object | None = None,
    ) -> None:
        if not manual:
            raise ValueError("manual must contain at least one proxy query")
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.manual = manual
        self.embedder = embedder or HashingTfidfEmbedder()
        queries = [entry.query for entry in manual]
        self.embedder.fit(queries)
        self._vectors = self._load_or_build_vectors(
            queries=queries,
            cache_path=Path(embedding_cache_path) if embedding_cache_path else None,
            force_rebuild=force_rebuild_embeddings,
            progress_callback=embedding_progress_callback,
        )

    def _load_or_build_vectors(
        self,
        queries: list[str],
        cache_path: Path | None,
        force_rebuild: bool,
        progress_callback: object | None,
    ) -> list[list[float]]:
        signature = {
            "version": 1,
            "embedder": embedder_signature(self.embedder),
            "manual_digest": text_digest([f"{entry.tool_name}\0{entry.query}" for entry in self.manual]),
            "count": len(self.manual),
        }
        if cache_path and cache_path.exists() and not force_rebuild:
            vectors = _load_vector_cache(cache_path, signature, len(self.manual))
            if vectors is not None:
                if progress_callback:
                    progress_callback(len(queries), len(queries))
                return vectors
        partial_path = cache_path.with_suffix(cache_path.suffix + ".partial.jsonl") if cache_path else None
        vectors: list[list[float]] = []
        start_index = 0
        if partial_path and partial_path.exists() and not force_rebuild:
            with partial_path.open(encoding="utf-8") as handle:
                first = handle.readline()
                try:
                    header = json.loads(first) if first.strip() else {}
                except json.JSONDecodeError:
                    header = {}
                if header.get("signature") == signature:
                    for line in handle:
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        vectors.append(item["embedding"])
                    start_index = len(vectors)
                    if progress_callback and start_index:
                        progress_callback(start_index, len(queries))
                else:
                    vectors = []
                    start_index = 0

        remaining = queries[start_index:]
        if remaining:
            if partial_path:
                partial_path.parent.mkdir(parents=True, exist_ok=True)
                mode = "a" if start_index else "w"
                with partial_path.open(mode, encoding="utf-8") as handle:
                    if not start_index:
                        handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")

                    def on_progress(done: int, total: int) -> None:
                        if progress_callback:
                            progress_callback(start_index + done, len(queries))

                    batch_size = max(1, getattr(self.embedder, "batch_size", len(remaining) or 1))
                    for offset in range(0, len(remaining), batch_size):
                        batch = remaining[offset : offset + batch_size]
                        batch_vectors = self.embedder.encode_many(batch)
                        for vector in batch_vectors:
                            vectors.append(vector)
                            handle.write(json.dumps({"embedding": vector}, ensure_ascii=False) + "\n")
                        handle.flush()
                        on_progress(min(offset + batch_size, len(remaining)), len(remaining))
            else:
                vectors = self.embedder.encode_many(queries, progress_callback=progress_callback)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if partial_path and partial_path.exists():
                partial_path.replace(cache_path)
            else:
                _write_vector_cache(cache_path, signature, vectors)
        return vectors

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


def _load_vector_cache(path: Path, signature: dict, expected_count: int) -> list[list[float]] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            if first.strip():
                header = json.loads(first)
                if isinstance(header, dict) and header.get("signature") == signature and "vectors" not in header:
                    vectors = [json.loads(line)["embedding"] for line in handle if line.strip()]
                    return vectors if len(vectors) == expected_count else None
    except Exception:
        pass

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("signature") != signature:
        return None
    vectors = payload.get("vectors", [])
    return vectors if isinstance(vectors, list) and len(vectors) == expected_count else None


def _write_vector_cache(path: Path, signature: dict, vectors: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
        for vector in vectors:
            handle.write(json.dumps({"embedding": vector}, ensure_ascii=False) + "\n")
