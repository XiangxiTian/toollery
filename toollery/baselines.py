from __future__ import annotations

import json
import logging
import math
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .embeddings import HashingTfidfEmbedder, LocalHFEmbedder, OpenAICompatibleEmbedder, cosine, embedder_signature, text_digest
from .llm import FinalSelector
from .schemas import ToolCall, ToolCandidate, ToolSpec
from .text import tokenize


FORBIDDEN_OUTPUT_PREFIXES = (
    Path("outputs/toollery_skillrouter_full_dp"),
    Path("outputs/toollery_skillrouter_full_dp/embeddings"),
)


@dataclass(frozen=True)
class RetrievalResult:
    tool: ToolSpec
    score: float
    document: str


@dataclass(frozen=True)
class BaselinePrediction:
    method_setting: str
    method_name: str
    benchmark: str
    sample_id: str
    query: str
    candidate_pool_size: int
    retrieved_candidates: list[str]
    correct_candidates: list[str]
    top_k_hit: bool | None
    final_prediction: str
    final_success: bool | None
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_latency_ms: float
    rerank_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float


class BaselineRetriever(Protocol):
    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        ...


@dataclass(frozen=True)
class EmbeddingConfig:
    backend: str = "openai-compatible"
    api_key: str | None = None
    base_url: str | None = None
    model: str = "text-embedding-3-small"
    dim: int = 1536
    timeout: int | None = None
    device: str | None = None
    batch_size: int = 8
    max_length: int = 512
    pooling: str = "last"
    dtype: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False


def ensure_safe_output_path(path: str | Path) -> None:
    resolved = Path(path).resolve()
    for prefix in FORBIDDEN_OUTPUT_PREFIXES:
        safe_prefix = prefix.resolve()
        try:
            resolved.relative_to(safe_prefix)
        except ValueError:
            continue
        raise ValueError(
            f"Refusing to write to {resolved}: this path is reserved for an in-progress background run."
        )


def read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_safe_output_path(path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    ensure_safe_output_path(path)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_generated_queries(path: str | Path | None) -> dict[str, list[str]]:
    if not path:
        return {}
    source = Path(path)
    if not source.exists():
        return {}
    if source.suffix == ".json":
        data = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "manual" in data:
            rows = data["manual"]
        elif isinstance(data, dict):
            return {str(key): [str(item) for item in value] for key, value in data.items() if isinstance(value, list)}
        else:
            rows = data
    else:
        rows = list(read_jsonl(source))
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        tool_name = row.get("tool_name") or row.get("skill_id") or row.get("name")
        query = str(row.get("query", "")).strip()
        if tool_name and query:
            out[str(tool_name)].append(query)
    return dict(out)


def normalize_generated_queries(
    source_path: str | Path,
    out_path: str | Path,
    accepted_only: bool = False,
) -> dict[str, list[str]]:
    rows = []
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl(source_path):
        if accepted_only and row.get("accepted") is not True:
            continue
        skill_id = row.get("skill_id") or row.get("tool_name")
        query = str(row.get("query", "")).strip()
        if not skill_id or not query:
            continue
        grouped[str(skill_id)].append(query)
        rows.append({"tool_name": str(skill_id), "query": query})
    write_jsonl(out_path, rows)
    return dict(grouped)


def build_tool_document(
    tool: ToolSpec,
    generated_queries: Iterable[str] | None = None,
    include_generated: bool = False,
) -> str:
    parts = [
        f"TOOL_NAME: {tool.name}",
        f"CATEGORY: {tool.category or ''}",
        f"DESCRIPTION: {tool.description}",
    ]
    if tool.parameters:
        parts.append("PARAMETERS_SCHEMA: " + json.dumps(tool.parameters, ensure_ascii=False, sort_keys=True))
    if include_generated:
        queries = [query for query in generated_queries or [] if query]
        if queries:
            parts.append("GENERATED_QUERIES:\n" + "\n".join(f"- {query}" for query in queries))
    return "\n".join(parts)


class BM25Retriever:
    def __init__(
        self,
        generated_queries: dict[str, list[str]] | None = None,
        include_generated: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.generated_queries = generated_queries or {}
        self.include_generated = include_generated
        self.k1 = k1
        self.b = b
        self._cache: dict[str, tuple[list[ToolSpec], list[str], list[list[str]], dict[str, int], float]] = {}

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        key = _tools_digest(tools, self.generated_queries if self.include_generated else None)
        cached = self._cache.get(key)
        if cached is None:
            documents = [
                build_tool_document(
                    tool,
                    self.generated_queries.get(tool.name, []),
                    include_generated=self.include_generated,
                )
                for tool in tools
            ]
            tokenized = [tokenize(document) for document in documents]
            dfs: Counter[str] = Counter()
            for tokens in tokenized:
                dfs.update(set(tokens))
            avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
            cached = (tools, documents, tokenized, dict(dfs), avgdl)
            self._cache[key] = cached
        cached_tools, documents, tokenized, dfs, avgdl = cached
        query_terms = tokenize(query)
        doc_count = max(len(cached_tools), 1)
        hits: list[RetrievalResult] = []
        for tool, document, tokens in zip(cached_tools, documents, tokenized):
            tf = Counter(tokens)
            score = 0.0
            dl = len(tokens)
            for term in query_terms:
                if term not in tf:
                    continue
                df = dfs.get(term, 0)
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                denom = tf[term] + self.k1 * (1.0 - self.b + self.b * dl / max(avgdl, 1e-9))
                score += idf * tf[term] * (self.k1 + 1.0) / denom
            hits.append(RetrievalResult(tool=tool, score=score, document=document))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class DenseRetriever:
    def __init__(
        self,
        embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder | None = None,
        generated_queries: dict[str, list[str]] | None = None,
        include_generated: bool = False,
        embedding_cache_path: str | Path | None = None,
        force_rebuild_embeddings: bool = False,
        embedding_progress_callback: object | None = None,
    ) -> None:
        self.embedder = embedder or HashingTfidfEmbedder()
        self.generated_queries = generated_queries or {}
        self.include_generated = include_generated
        self.embedding_cache_path = Path(embedding_cache_path) if embedding_cache_path else None
        self.force_rebuild_embeddings = force_rebuild_embeddings
        self.embedding_progress_callback = embedding_progress_callback
        self._cache: dict[str, tuple[list[ToolSpec], list[str], list[list[float]]]] = {}
        self._document_vector_cache: dict[str, list[float]] = {}
        self._document_cache_loaded = False

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        if not isinstance(self.embedder, HashingTfidfEmbedder):
            return self._retrieve_with_global_document_cache(query, tools, top_k)

        key = _tools_digest(tools, self.generated_queries if self.include_generated else None)
        cached = self._cache.get(key)
        if cached is None:
            documents = [
                build_tool_document(
                    tool,
                    self.generated_queries.get(tool.name, []),
                    include_generated=self.include_generated,
                )
                for tool in tools
            ]
            self.embedder.fit(documents)
            vectors = _load_or_build_document_vectors(
                documents=documents,
                embedder=self.embedder,
                cache_path=_resolve_embedding_cache_path(self.embedding_cache_path, key),
                cache_key=key,
                force_rebuild=self.force_rebuild_embeddings,
                progress_callback=self.embedding_progress_callback,
            )
            cached = (tools, documents, vectors)
            self._cache[key] = cached
        cached_tools, documents, vectors = cached
        query_vector = self.embedder.encode(query)
        hits = [
            RetrievalResult(tool=tool, score=cosine(query_vector, vector), document=document)
            for tool, document, vector in zip(cached_tools, documents, vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _retrieve_with_global_document_cache(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        documents = [
            build_tool_document(
                tool,
                self.generated_queries.get(tool.name, []),
                include_generated=self.include_generated,
            )
            for tool in tools
        ]
        self.embedder.fit(documents)
        vectors = _load_or_build_global_document_vectors(
            documents=documents,
            embedder=self.embedder,
            cache_path=_resolve_global_embedding_cache_path(self.embedding_cache_path),
            force_rebuild=self.force_rebuild_embeddings and not self._document_cache_loaded,
            vector_cache=self._document_vector_cache,
            cache_loaded=self._document_cache_loaded,
            progress_callback=self.embedding_progress_callback,
        )
        self._document_cache_loaded = True
        query_vector = self.embedder.encode(query)
        hits = [
            RetrievalResult(tool=tool, score=cosine(query_vector, vector), document=document)
            for tool, document, vector in zip(tools, documents, vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class NameDescriptionTfidfRAGRetriever:
    """Legacy RAG-style baseline over raw or query-augmented tool documents."""

    def __init__(
        self,
        generated_queries: dict[str, list[str]] | None = None,
        include_generated: bool = False,
    ) -> None:
        self.generated_queries = generated_queries or {}
        self.include_generated = include_generated
        self._cache: dict[str, tuple[list[ToolSpec], list[str], HashingTfidfEmbedder, list[list[float]]]] = {}

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        key = _tools_digest(tools, self.generated_queries if self.include_generated else None)
        cached = self._cache.get(key)
        if cached is None:
            documents = [
                build_tool_document(
                    tool,
                    self.generated_queries.get(tool.name, []),
                    include_generated=self.include_generated,
                )
                for tool in tools
            ]
            embedder = HashingTfidfEmbedder()
            embedder.fit(documents)
            vectors = [embedder.encode(document) for document in documents]
            cached = (tools, documents, embedder, vectors)
            self._cache[key] = cached
        cached_tools, documents, embedder, vectors = cached
        query_vector = embedder.encode(query)
        hits = [
            RetrievalResult(tool=tool, score=cosine(query_vector, vector), document=document)
            for tool, document, vector in zip(cached_tools, documents, vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class RAGAnythingNameDescriptionRetriever:
    """RAG-Anything/LightRAG-backed raw or query-augmented baseline retriever."""

    def __init__(
        self,
        working_dir: str | Path = ".raganything_baseline",
        embedding_config: EmbeddingConfig | None = None,
        generated_queries: dict[str, list[str]] | None = None,
        include_generated: bool = False,
    ) -> None:
        self.working_dir = Path(working_dir).resolve()
        self.embedding_config = embedding_config or EmbeddingConfig()
        self.generated_queries = generated_queries or {}
        self.include_generated = include_generated
        self._loop: Any | None = None
        self._stores: dict[str, tuple[Any, Any]] = {}
        self._embedding_func: Any | None = None

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        if self._loop is None:
            print("[raganything] loading LightRAG backend", file=sys.stderr, flush=True)
        try:
            import asyncio
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag.utils import EmbeddingFunc, setup_logger
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "RAG-Anything backend requires optional dependencies. Install with "
                "`pip install raganything lightrag-hku` and set OpenAI-compatible "
                "environment variables for LightRAG."
            ) from exc

        if self._loop is None:
            self._loop = asyncio.new_event_loop()

        return self._loop.run_until_complete(
            self._retrieve_async(
                query=query,
                tools=tools,
                top_k=top_k,
                LightRAG=LightRAG,
                initialize_pipeline_status=initialize_pipeline_status,
                EmbeddingFunc=EmbeddingFunc,
                setup_logger=setup_logger,
            )
        )

    def close(self) -> None:
        if self._loop is None or self._loop.is_closed():
            self._stores.clear()
            return
        self._loop.run_until_complete(self._close_async())
        pending = [task for task in _all_tasks(self._loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(_gather_tasks(pending))
        self._loop.close()
        self._loop = None
        self._stores.clear()

    async def _retrieve_async(
        self,
        *,
        query: str,
        tools: list[ToolSpec],
        top_k: int,
        LightRAG: Any,
        initialize_pipeline_status: Any,
        EmbeddingFunc: Any,
        setup_logger: Any,
    ) -> list[RetrievalResult]:
        inserted_key = "_".join(
            [
                _tools_digest(tools, self.generated_queries if self.include_generated else None),
                _embedding_config_digest(self.embedding_config),
            ]
        )
        working_dir = self.working_dir / inserted_key
        marker = working_dir / ".inserted"

        store = self._stores.get(inserted_key)
        if store is None and working_dir.exists():
            repair_reason = _lightrag_cache_rebuild_reason(working_dir, marker)
            if repair_reason:
                _move_lightrag_cache_aside(working_dir, repair_reason)

        if store is None:
            working_dir.mkdir(parents=True, exist_ok=True)
            print(f"[raganything] vector store: {working_dir}", file=sys.stderr, flush=True)
            setup_logger("lightrag", level="WARNING")
            logging.getLogger("nano-vectordb").setLevel(logging.WARNING)
            logging.getLogger("openai").setLevel(logging.WARNING)
            logging.getLogger("httpx").setLevel(logging.WARNING)
            lightrag = LightRAG(
                working_dir=str(working_dir),
                llm_model_func=_unused_llm,
                llm_model_name="unused-for-chunk-retrieval",
                embedding_func=self._get_embedding_func(EmbeddingFunc),
                embedding_batch_num=max(1, self.embedding_config.batch_size),
                embedding_func_max_async=2,
                default_embedding_timeout=max(self.embedding_config.timeout or 0, 600),
            )
            setattr(lightrag, "_toollery_generated_queries", self.generated_queries)
            setattr(lightrag, "_toollery_include_generated", self.include_generated)
            await lightrag.initialize_storages()
            await initialize_pipeline_status()
            store = (lightrag, None)
            self._stores[inserted_key] = store
        lightrag, _rag = store

        if not marker.exists():
            (working_dir / ".building").write_text("building chunk vectors\n", encoding="utf-8")
            await _insert_chunk_vectors(
                lightrag,
                tools,
                inserted_key,
                batch_size=max(1, self.embedding_config.batch_size),
            )
            if not _has_indexed_chunks(working_dir):
                raise RuntimeError(
                    f"RAG-Anything inserted no text chunks into {working_dir}. "
                    "Remove that directory and rerun after checking the input documents."
                )
            marker.write_text("ok", encoding="utf-8")
            (working_dir / ".building").unlink(missing_ok=True)

        chunk_hits = await lightrag.chunks_vdb.query(query, top_k=top_k)
        names_in_order = []
        scores_by_name: dict[str, float] = {}
        for hit in chunk_hits:
            content = str(hit.get("content", ""))
            for name in _extract_tool_names_from_context(content):
                names_in_order.append(name)
                scores_by_name.setdefault(name, float(hit.get("distance", 0.0)))
        tools_by_name = {tool.name: tool for tool in tools}
        hits: list[RetrievalResult] = []
        seen: set[str] = set()
        for rank, name in enumerate(names_in_order):
            if name in seen or name not in tools_by_name:
                continue
            seen.add(name)
            tool = tools_by_name[name]
            hits.append(
                RetrievalResult(
                    tool=tool,
                    score=scores_by_name.get(name, 1.0 / (rank + 1)),
                    document=build_tool_document(
                        tool,
                        self.generated_queries.get(tool.name, []),
                        include_generated=self.include_generated,
                    ),
                )
            )
            if len(hits) >= top_k:
                break
        if len(hits) < top_k:
            fallback = NameDescriptionTfidfRAGRetriever(
                generated_queries=self.generated_queries,
                include_generated=self.include_generated,
            ).retrieve(query, tools, top_k)
            for hit in fallback:
                if hit.tool.name not in seen:
                    hits.append(hit)
                if len(hits) >= top_k:
                    break
        return hits

    def _get_embedding_func(self, EmbeddingFunc: Any) -> Any:
        if self._embedding_func is None:
            self._embedding_func = _make_embedding_func(EmbeddingFunc, self.embedding_config)
        return self._embedding_func

    async def _close_async(self) -> None:
        for lightrag, _rag in self._stores.values():
            finalize = getattr(lightrag, "finalize_storages", None)
            if finalize is not None:
                await finalize()


def make_rag_retriever(
    backend: str,
    working_dir: str | Path | None = None,
    embedding_config: EmbeddingConfig | None = None,
    generated_queries: dict[str, list[str]] | None = None,
    include_generated: bool = False,
) -> BaselineRetriever:
    if backend == "tfidf":
        return NameDescriptionTfidfRAGRetriever(
            generated_queries=generated_queries,
            include_generated=include_generated,
        )
    if backend in {"raganything", "lightrag"}:
        return RAGAnythingNameDescriptionRetriever(
            working_dir or ".raganything_baseline",
            embedding_config=embedding_config,
            generated_queries=generated_queries,
            include_generated=include_generated,
        )
    raise ValueError(f"unknown RAG backend: {backend}")


def tool_to_rag_document(
    tool: ToolSpec,
    generated_queries: Iterable[str] | None = None,
    include_generated: bool = False,
) -> str:
    return build_tool_document(
        tool,
        generated_queries=generated_queries,
        include_generated=include_generated,
    )


def run_retrieval_baseline(
    *,
    query: str,
    tools: list[ToolSpec],
    correct_tools: list[str],
    retriever: BaselineRetriever,
    method_setting: str,
    method_name: str,
    benchmark: str,
    sample_id: str,
    top_k: int,
    selector: FinalSelector | None = None,
) -> BaselinePrediction:
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    hits = retriever.retrieve(query, tools, min(top_k, len(tools)))
    retrieval_latency_ms = _elapsed_ms(retrieval_start)
    retrieved = [hit.tool.name for hit in hits]
    top_k_hit = bool(set(retrieved) & set(correct_tools)) if correct_tools else None
    llm_latency_ms = 0.0
    final_prediction = retrieved[0] if retrieved else (tools[0].name if tools else "")
    if selector is not None:
        llm_start = time.perf_counter()
        call = selector.choose_tool(query, [hit.tool for hit in hits])
        llm_latency_ms = _elapsed_ms(llm_start)
        final_prediction = call.tool_name
    final_success = final_prediction in correct_tools if correct_tools else None
    prompt_tokens = estimate_prompt_tokens(query, [hit.tool for hit in hits])
    return BaselinePrediction(
        method_setting=method_setting,
        method_name=method_name,
        benchmark=benchmark,
        sample_id=sample_id,
        query=query,
        candidate_pool_size=len(tools),
        retrieved_candidates=retrieved,
        correct_candidates=correct_tools,
        top_k_hit=top_k_hit,
        final_prediction=final_prediction,
        final_success=final_success,
        prompt_tokens=prompt_tokens,
        completion_tokens=None,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=0.0,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=_elapsed_ms(start),
    )


def run_full_context_llm_baseline(
    *,
    query: str,
    tools: list[ToolSpec],
    correct_tools: list[str],
    selector: FinalSelector,
    method_name: str,
    benchmark: str,
    sample_id: str,
) -> BaselinePrediction:
    start = time.perf_counter()
    llm_start = time.perf_counter()
    call: ToolCall = selector.choose_tool(query, tools)
    llm_latency_ms = _elapsed_ms(llm_start)
    retrieved = [tool.name for tool in tools]
    final_success = call.tool_name in correct_tools if correct_tools else None
    top_k_hit = bool(set(retrieved) & set(correct_tools)) if correct_tools else None
    return BaselinePrediction(
        method_setting="raw_baseline",
        method_name=method_name,
        benchmark=benchmark,
        sample_id=sample_id,
        query=query,
        candidate_pool_size=len(tools),
        retrieved_candidates=retrieved,
        correct_candidates=correct_tools,
        top_k_hit=top_k_hit,
        final_prediction=call.tool_name,
        final_success=final_success,
        prompt_tokens=estimate_prompt_tokens(query, tools),
        completion_tokens=None,
        retrieval_latency_ms=0.0,
        rerank_latency_ms=0.0,
        llm_latency_ms=llm_latency_ms,
        total_latency_ms=_elapsed_ms(start),
    )


def estimate_prompt_tokens(query: str, tools: list[ToolSpec]) -> int:
    text = query + "\n" + "\n".join(build_tool_document(tool) for tool in tools)
    return max(1, len(text) // 4)


def summarize_predictions(predictions: list[BaselinePrediction]) -> dict[str, Any]:
    evaluated_topk = [row for row in predictions if row.top_k_hit is not None]
    evaluated_final = [row for row in predictions if row.final_success is not None]
    return {
        "count": len(predictions),
        "top_k_hit": _mean([1.0 if row.top_k_hit else 0.0 for row in evaluated_topk]),
        "final_success": _mean([1.0 if row.final_success else 0.0 for row in evaluated_final]),
        "efficiency": {
            "avg_prompt_tokens": _mean([row.prompt_tokens for row in predictions if row.prompt_tokens is not None]),
            "avg_retrieval_latency_ms": _mean([row.retrieval_latency_ms for row in predictions]),
            "avg_rerank_latency_ms": _mean([row.rerank_latency_ms for row in predictions]),
            "avg_llm_latency_ms": _mean([row.llm_latency_ms for row in predictions]),
            "avg_total_latency_ms": _mean([row.total_latency_ms for row in predictions]),
        },
    }


def _load_or_build_document_vectors(
    *,
    documents: list[str],
    embedder: HashingTfidfEmbedder | OpenAICompatibleEmbedder | LocalHFEmbedder,
    cache_path: Path | None,
    cache_key: str,
    force_rebuild: bool,
    progress_callback: object | None,
) -> list[list[float]]:
    signature = {
        "version": 1,
        "kind": "baseline_document_embeddings",
        "cache_key": cache_key,
        "embedder": embedder_signature(embedder),
        "document_digest": text_digest(documents),
        "count": len(documents),
    }
    if cache_path and cache_path.exists() and not force_rebuild:
        vectors = _load_vector_cache(cache_path, signature, len(documents))
        if vectors is not None:
            if progress_callback:
                progress_callback(len(documents), len(documents), "cache_hit")
            return vectors

    partial_path = cache_path.with_suffix(cache_path.suffix + ".partial.jsonl") if cache_path else None
    vectors: list[list[float]] = []
    start_index = 0
    if partial_path and partial_path.exists() and not force_rebuild:
        vectors = _load_vector_cache(partial_path, signature, len(documents), allow_partial=True) or []
        start_index = len(vectors)
        if progress_callback and start_index:
            progress_callback(start_index, len(documents), "resumed")

    remaining = documents[start_index:]
    if remaining:
        batch_size = max(1, int(getattr(embedder, "batch_size", 128) or 128))
        if partial_path:
            ensure_safe_output_path(partial_path)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if start_index else "w"
            with partial_path.open(mode, encoding="utf-8") as handle:
                if not start_index:
                    handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
                for offset in range(0, len(remaining), batch_size):
                    batch = remaining[offset : offset + batch_size]
                    batch_vectors = embedder.encode_many(batch)
                    for vector in batch_vectors:
                        vectors.append(vector)
                        handle.write(json.dumps({"embedding": vector}, ensure_ascii=False) + "\n")
                    handle.flush()
                    if progress_callback:
                        progress_callback(min(start_index + offset + len(batch), len(documents)), len(documents), "building")
        else:
            vectors = []
            for offset in range(0, len(documents), batch_size):
                batch = documents[offset : offset + batch_size]
                vectors.extend(embedder.encode_many(batch))
                if progress_callback:
                    progress_callback(min(offset + len(batch), len(documents)), len(documents), "building")

    if cache_path:
        ensure_safe_output_path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if partial_path and partial_path.exists():
            partial_path.replace(cache_path)
        else:
            _write_vector_cache(cache_path, signature, vectors)
    return vectors


def _load_or_build_global_document_vectors(
    *,
    documents: list[str],
    embedder: OpenAICompatibleEmbedder | LocalHFEmbedder,
    cache_path: Path | None,
    force_rebuild: bool,
    vector_cache: dict[str, list[float]],
    cache_loaded: bool,
    progress_callback: object | None,
) -> list[list[float]]:
    signature = {
        "version": 1,
        "kind": "global_baseline_document_embeddings",
        "embedder": embedder_signature(embedder),
    }
    if cache_path and not cache_loaded and cache_path.exists() and not force_rebuild:
        vector_cache.update(_load_global_vector_cache(cache_path, signature))

    document_ids = [_document_id(document) for document in documents]
    unique_missing: list[tuple[str, str]] = []
    seen_missing: set[str] = set()
    for document_id, document in zip(document_ids, documents):
        if document_id in vector_cache or document_id in seen_missing:
            continue
        seen_missing.add(document_id)
        unique_missing.append((document_id, document))

    if unique_missing:
        cached_for_current = len({document_id for document_id in document_ids if document_id in vector_cache})
        if progress_callback and cached_for_current:
            progress_callback(cached_for_current, len(set(document_ids)), "resumed")
        batch_size = max(1, int(getattr(embedder, "batch_size", 128) or 128))
        if cache_path:
            ensure_safe_output_path(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "w" if force_rebuild or not cache_path.exists() else "a"
            with cache_path.open(mode, encoding="utf-8") as handle:
                if mode == "w":
                    handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
                for offset in range(0, len(unique_missing), batch_size):
                    batch = unique_missing[offset : offset + batch_size]
                    batch_ids = [item[0] for item in batch]
                    batch_documents = [item[1] for item in batch]
                    batch_vectors = embedder.encode_many(batch_documents)
                    for document_id, vector in zip(batch_ids, batch_vectors):
                        vector_cache[document_id] = vector
                        handle.write(
                            json.dumps({"document_id": document_id, "embedding": vector}, ensure_ascii=False) + "\n"
                        )
                    handle.flush()
                    if progress_callback:
                        done = cached_for_current + min(offset + len(batch), len(unique_missing))
                        progress_callback(done, len(set(document_ids)), "building")
        else:
            for offset in range(0, len(unique_missing), batch_size):
                batch = unique_missing[offset : offset + batch_size]
                batch_vectors = embedder.encode_many([item[1] for item in batch])
                for (document_id, _), vector in zip(batch, batch_vectors):
                    vector_cache[document_id] = vector
                if progress_callback:
                    done = min(offset + len(batch), len(unique_missing))
                    progress_callback(done, len(unique_missing), "building")
    elif progress_callback and documents:
        progress_callback(len(set(document_ids)), len(set(document_ids)), "cache_hit")

    return [vector_cache[document_id] for document_id in document_ids]


def _resolve_embedding_cache_path(cache_path: Path | None, cache_key: str) -> Path | None:
    if cache_path is None:
        return None
    if cache_path.suffix:
        return cache_path.with_name(f"{cache_path.stem}.{cache_key}{cache_path.suffix}")
    return cache_path / f"{cache_key}.jsonl"


def _resolve_global_embedding_cache_path(cache_path: Path | None) -> Path | None:
    if cache_path is None:
        return None
    if cache_path.suffix:
        return cache_path
    return cache_path / "global_documents.jsonl"


def _load_vector_cache(
    path: Path,
    signature: dict[str, Any],
    expected_count: int,
    allow_partial: bool = False,
) -> list[list[float]] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            header = json.loads(first) if first.strip() else {}
            if not isinstance(header, dict) or header.get("signature") != signature:
                return None
            vectors = [json.loads(line)["embedding"] for line in handle if line.strip()]
            if allow_partial:
                return vectors if len(vectors) <= expected_count else None
            return vectors if len(vectors) == expected_count else None
    except Exception:
        return None


def _load_global_vector_cache(path: Path, signature: dict[str, Any]) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {}
    try:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline()
            header = json.loads(first) if first.strip() else {}
            if not isinstance(header, dict) or header.get("signature") != signature:
                return {}
            for line in handle:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                document_id = item.get("document_id")
                embedding = item.get("embedding")
                if isinstance(document_id, str) and isinstance(embedding, list):
                    vectors[document_id] = embedding
    except Exception:
        return {}
    return vectors


def _write_vector_cache(path: Path, signature: dict[str, Any], vectors: list[list[float]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"signature": signature}, ensure_ascii=False) + "\n")
        for vector in vectors:
            handle.write(json.dumps({"embedding": vector}, ensure_ascii=False) + "\n")


def _document_id(document: str) -> str:
    return text_digest([document])


def _sanitize_lightrag_content(content: str) -> str:
    """Prevent tiktoken from treating literal special-token strings as control tokens."""

    return re.sub(r"<\|([^|\r\n]{1,100})\|>", r"< |\1| >", content)


def _extract_tool_names_from_context(context: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"TOOL_NAME:\s*([^\n\r]+)", context)]


async def _insert_chunk_vectors(
    lightrag: Any,
    tools: list[ToolSpec],
    inserted_key: str,
    batch_size: int = 256,
) -> None:
    chunks: dict[str, dict[str, Any]] = {}
    full_docs: dict[str, dict[str, Any]] = {}
    file_path = f"toollery_tools_{inserted_key}.txt"
    working_dir = getattr(lightrag, "working_dir", "")
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)
    existing_chunk_ids = _indexed_chunk_ids(Path(working_dir)) if working_dir else set()
    print(
        f"[raganything] preparing {len(tools)} name-description chunks in {working_dir}",
        file=sys.stderr,
        flush=True,
    )
    for index, tool in enumerate(tools):
        generated = getattr(lightrag, "_toollery_generated_queries", {})
        include_generated = bool(getattr(lightrag, "_toollery_include_generated", False))
        content = build_tool_document(
            tool,
            generated.get(tool.name, []) if isinstance(generated, dict) else [],
            include_generated=include_generated,
        )
        content = _sanitize_lightrag_content(content)
        doc_id = f"toollery-doc-{inserted_key}-{index}"
        chunk_id = f"toollery-chunk-{inserted_key}-{index}"
        full_docs[doc_id] = {"content": content, "file_path": file_path}
        if chunk_id not in existing_chunk_ids:
            chunks[chunk_id] = {
                "content": content,
                "full_doc_id": doc_id,
                "tokens": len(lightrag.tokenizer.encode(content)),
                "chunk_order_index": 0,
                "file_path": file_path,
            }

    await lightrag.full_docs.upsert(full_docs)
    if chunks:
        await lightrag.text_chunks.upsert(chunks)
    _ensure_lightrag_working_dir(lightrag)
    await lightrag.full_docs.index_done_callback()
    _ensure_lightrag_working_dir(lightrag)
    await lightrag.text_chunks.index_done_callback()
    if not chunks:
        print("[raganything] chunk vector store already complete", file=sys.stderr, flush=True)
        return
    if existing_chunk_ids:
        print(
            f"[raganything] resuming chunk vector store; {len(existing_chunk_ids)} chunks already embedded",
            file=sys.stderr,
            flush=True,
        )
    print(
        f"[raganything] persisted text metadata; embedding chunks in batches of {batch_size}",
        file=sys.stderr,
        flush=True,
    )

    chunk_items = list(chunks.items())
    total = len(chunk_items)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = dict(chunk_items[start:end])
        await lightrag.chunks_vdb.upsert(batch)
        _ensure_lightrag_working_dir(lightrag)
        await lightrag.chunks_vdb.index_done_callback()
        _render_embedding_bar("raganything", "embedding text chunks", end, total, final=end == total)


def _ensure_lightrag_working_dir(lightrag: Any) -> None:
    working_dir = getattr(lightrag, "working_dir", "")
    if working_dir:
        Path(working_dir).mkdir(parents=True, exist_ok=True)


def _render_embedding_bar(prefix: str, label: str, done: int, total: int, final: bool = False) -> None:
    width = 28
    if total <= 0:
        line = f"[{prefix}] {label}: 0/0"
    else:
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        line = f"[{prefix}] {label}: [{bar}] {done}/{total} {ratio * 100:5.1f}%"
    print(line, file=sys.stderr, end="\n" if final else "\r", flush=True)


async def _unused_llm(*_: Any, **__: Any) -> str:
    return ""


def _make_embedding_func(EmbeddingFunc: Any, config: EmbeddingConfig) -> Any:
    backend = config.backend.lower().replace("_", "-")
    if backend in {"local", "local-hf", "hf", "huggingface", "transformers"}:
        return _make_local_hf_embedding_func(EmbeddingFunc, config)
    return _make_openai_embedding_func(EmbeddingFunc, config)


def _make_openai_embedding_func(EmbeddingFunc: Any, config: EmbeddingConfig) -> Any:
    async def embed(texts: list[str], **_: Any) -> Any:
        import numpy as np
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )
        async with client:
            response = await client.embeddings.create(
                model=config.model,
                input=texts,
                dimensions=config.dim,
                encoding_format="float",
            )
        return np.array([item.embedding for item in response.data], dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=config.dim,
        max_token_size=8192,
        func=embed,
        model_name=config.model,
        supports_asymmetric=True,
    )


def _make_local_hf_embedding_func(EmbeddingFunc: Any, config: EmbeddingConfig) -> Any:
    import numpy as np

    embedder = LocalHFEmbedder(
        model=config.model,
        device=config.device,
        batch_size=config.batch_size,
        max_length=config.max_length,
        pooling=config.pooling,
        dtype=config.dtype,
        trust_remote_code=config.trust_remote_code,
        local_files_only=config.local_files_only,
    )
    probe = embedder.encode("dimension probe")
    embedding_dim = len(probe)

    async def embed(texts: list[str], **_: Any) -> Any:
        return np.array(embedder.encode_many(texts), dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=config.max_length,
        func=embed,
        model_name=config.model,
        supports_asymmetric=True,
    )


def _all_tasks(loop: Any) -> set[Any]:
    import asyncio

    return asyncio.all_tasks(loop)


async def _gather_tasks(tasks: list[Any]) -> None:
    import asyncio

    await asyncio.gather(*tasks, return_exceptions=True)


def _has_indexed_chunks(working_dir: Path) -> bool:
    return bool(_indexed_chunk_ids(working_dir))


def _lightrag_cache_rebuild_reason(working_dir: Path, marker: Path) -> str | None:
    if (working_dir / ".building").exists():
        if _has_indexed_chunks(working_dir):
            return None
        return "previous build did not finish before any chunk vectors were indexed"
    if marker.exists() and not _has_indexed_chunks(working_dir):
        return "inserted marker exists but chunk vector index is missing or corrupt"
    if not marker.exists() and any(working_dir.glob("*.json")):
        return "partial cache files exist without an inserted marker"
    return None


def _move_lightrag_cache_aside(working_dir: Path, reason: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    target = working_dir.with_name(f"{working_dir.name}.corrupt-{timestamp}")
    suffix = 1
    while target.exists():
        target = working_dir.with_name(f"{working_dir.name}.corrupt-{timestamp}-{suffix}")
        suffix += 1
    print(
        f"[raganything] ignoring incomplete/corrupt vector store ({reason}); "
        f"moving {working_dir} to {target} and rebuilding",
        file=sys.stderr,
        flush=True,
    )
    shutil.move(str(working_dir), str(target))


def _indexed_chunk_ids(working_dir: Path) -> set[str]:
    chunks_path = working_dir / "vdb_chunks.json"
    if not chunks_path.exists():
        return set()
    try:
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            return {str(item["__id__"]) for item in data["data"] if isinstance(item, dict) and "__id__" in item}
        if isinstance(data.get("storage"), list):
            return {str(item["__id__"]) for item in data["storage"] if isinstance(item, dict) and "__id__" in item}
        ids: set[str] = set()
        for value in data.values():
            if isinstance(value, list):
                ids.update(str(item["__id__"]) for item in value if isinstance(item, dict) and "__id__" in item)
        return ids
    if isinstance(data, list):
        return {str(item["__id__"]) for item in data if isinstance(item, dict) and "__id__" in item}
    return set()


def _mean(values: list[float | int]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _tools_digest(tools: list[ToolSpec], generated_queries: dict[str, list[str]] | None = None) -> str:
    import hashlib

    digest = hashlib.sha1()
    for tool in tools:
        digest.update(build_tool_document(tool, generated_queries.get(tool.name, []) if generated_queries else [], bool(generated_queries)).encode("utf-8"))
        digest.update(b"\0\0")
    return digest.hexdigest()[:16]


def _embedding_config_digest(config: EmbeddingConfig) -> str:
    import hashlib

    payload = {
        "backend": config.backend,
        "base_url": config.base_url,
        "model": config.model,
        "dim": config.dim,
        "device": config.device,
        "max_length": config.max_length,
        "pooling": config.pooling,
        "dtype": config.dtype,
        "local_files_only": config.local_files_only,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]
