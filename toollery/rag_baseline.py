from __future__ import annotations

import json
import re
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .bfcl import BFCLScalePrediction, BFCLScaledSample
from .baselines import build_tool_document
from .embeddings import HashingTfidfEmbedder, LocalHFEmbedder, cosine
from .schemas import ToolSpec
from .skills import SkillScalePrediction, SkillScaledSample


@dataclass(frozen=True)
class RAGToolHit:
    tool: ToolSpec
    score: float
    document: str


class ToolRAGRetriever(Protocol):
    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RAGToolHit]:
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


class NameDescriptionTfidfRAGRetriever:
    """Simple RAG-style retriever over raw or query-augmented tool documents."""

    def __init__(
        self,
        generated_queries: dict[str, list[str]] | None = None,
        include_generated: bool = False,
    ) -> None:
        self._cache: dict[str, tuple[list[ToolSpec], list[str], HashingTfidfEmbedder, list[list[float]]]] = {}
        self.generated_queries = generated_queries or {}
        self.include_generated = include_generated

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RAGToolHit]:
        key = _tools_digest(tools, self.generated_queries if self.include_generated else None)
        cached = self._cache.get(key)
        if cached is None:
            documents = [
                tool_to_rag_document(
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
            RAGToolHit(tool=tool, score=cosine(query_vector, vector), document=document)
            for tool, document, vector in zip(cached_tools, documents, vectors)
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class RAGAnythingNameDescriptionRetriever:
    """RAG-Anything/LightRAG-backed retriever for the name-description baseline.

    RAG-Anything is optimized for end-to-end multimodal RAG, not a small in-memory
    ranking API. For this comparison we use it in text-only mode by inserting one
    compact document per tool and asking LightRAG for retrieval context only, then
    recover tool ids from the returned context order.
    """

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

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RAGToolHit]:
        if self._loop is None:
            print("[raganything] loading LightRAG backend", file=sys.stderr, flush=True)
        try:
            import asyncio
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag.utils import EmbeddingFunc, setup_logger
        except ImportError as exc:  # pragma: no cover - depends on optional package
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
    ) -> list[RAGToolHit]:
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
                embedding_func=_make_embedding_func(
                    EmbeddingFunc,
                    self.embedding_config,
                ),
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
            await _insert_chunk_vectors(lightrag, tools, inserted_key)
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
        hits: list[RAGToolHit] = []
        seen: set[str] = set()
        for rank, name in enumerate(names_in_order):
            if name in seen or name not in tools_by_name:
                continue
            seen.add(name)
            tool = tools_by_name[name]
            hits.append(
                RAGToolHit(
                    tool=tool,
                    score=scores_by_name.get(name, 1.0 / (rank + 1)),
                    document=tool_to_rag_document(
                        tool,
                        self.generated_queries.get(tool.name, []),
                        include_generated=self.include_generated,
                    ),
                )
            )
            if len(hits) >= top_k:
                break
        if len(hits) < top_k:
            fallback = NameDescriptionTfidfRAGRetriever().retrieve(query, tools, top_k)
            for hit in fallback:
                if hit.tool.name not in seen:
                    hits.append(hit)
                if len(hits) >= top_k:
                    break
        return hits

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
) -> ToolRAGRetriever:
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


def run_bfcl_rag_scaled_samples(
    scaled_samples: list[BFCLScaledSample],
    top_k: int = 3,
    retriever: ToolRAGRetriever | None = None,
) -> list[BFCLScalePrediction]:
    retriever = retriever or NameDescriptionTfidfRAGRetriever()
    predictions: list[BFCLScalePrediction] = []
    for sample in scaled_samples:
        hits = retriever.retrieve(sample.query, sample.tools, min(top_k, len(sample.tools)))
        retrieved_tools = [hit.tool.name for hit in hits]
        predicted_tool = retrieved_tools[0] if retrieved_tools else (sample.tools[0].name if sample.tools else "")
        top_k_hit = bool(set(retrieved_tools) & set(sample.correct_tools))
        predictions.append(
            BFCLScalePrediction(
                candidate_size=sample.candidate_size,
                sample_id=sample.sample_id,
                query=sample.query,
                predicted_tool=predicted_tool,
                retrieved_tools=retrieved_tools,
                scale_candidate_pool=[tool.name for tool in sample.tools],
                correct_tools=sample.correct_tools,
                is_correct=top_k_hit if sample.correct_tools else None,
            )
        )
    return predictions


def run_skill_rag_scaled_samples(
    scaled_samples: list[SkillScaledSample],
    top_k: int = 3,
    retriever: ToolRAGRetriever | None = None,
) -> list[SkillScalePrediction]:
    retriever = retriever or NameDescriptionTfidfRAGRetriever()
    predictions: list[SkillScalePrediction] = []
    for sample in scaled_samples:
        hits = retriever.retrieve(sample.query, sample.tools, min(top_k, len(sample.tools)))
        retrieved_tools = [hit.tool.name for hit in hits]
        predicted_tool = retrieved_tools[0] if retrieved_tools else (sample.tools[0].name if sample.tools else "")
        top_k_hit = bool(set(retrieved_tools) & set(sample.correct_tools))
        predictions.append(
            SkillScalePrediction(
                candidate_size=sample.candidate_size,
                sample_id=sample.sample_id,
                query=sample.query,
                predicted_tool=predicted_tool,
                retrieved_tools=retrieved_tools,
                scale_candidate_pool=[tool.name for tool in sample.tools],
                correct_tools=sample.correct_tools,
                is_correct=top_k_hit if sample.correct_tools else None,
            )
        )
    return predictions


def compare_scale_predictions(
    toollery_predictions: Iterable[Any],
    rag_predictions: Iterable[Any],
) -> list[dict[str, Any]]:
    by_key = {
        (prediction.candidate_size, prediction.sample_id): prediction
        for prediction in toollery_predictions
    }
    rows: list[dict[str, Any]] = []
    for rag in rag_predictions:
        toollery = by_key.get((rag.candidate_size, rag.sample_id))
        if toollery is None:
            continue
        correct_tools = set(rag.correct_tools)
        rows.append(
            {
                "candidate_size": rag.candidate_size,
                "sample_id": rag.sample_id,
                "query": rag.query,
                "correct_tools": rag.correct_tools,
                "toollery_predicted_tool": toollery.predicted_tool,
                "toollery_retrieved_tools": toollery.retrieved_tools,
                "toollery_is_correct": toollery.is_correct,
                "toollery_top_k_hit": bool(set(toollery.retrieved_tools) & correct_tools),
                "rag_predicted_tool": rag.predicted_tool,
                "rag_retrieved_tools": rag.retrieved_tools,
                "rag_is_correct": rag.is_correct,
                "rag_top_k_hit": bool(set(rag.retrieved_tools) & correct_tools),
            }
        )
    return rows


def save_comparison(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_bfcl_scale_predictions(path: str | Path) -> list[BFCLScalePrediction]:
    return [
        BFCLScalePrediction(
            candidate_size=int(item["candidate_size"]),
            sample_id=str(item["sample_id"]),
            query=str(item["query"]),
            predicted_tool=str(item["predicted_tool"]),
            retrieved_tools=[str(name) for name in item.get("retrieved_tools", [])],
            scale_candidate_pool=[str(name) for name in item.get("scale_candidate_pool", [])],
            correct_tools=[str(name) for name in item.get("correct_tools", [])],
            is_correct=item.get("is_correct"),
        )
        for item in _read_jsonl(path)
    ]


def load_skill_scale_predictions(path: str | Path) -> list[SkillScalePrediction]:
    return [
        SkillScalePrediction(
            candidate_size=int(item["candidate_size"]),
            sample_id=str(item["sample_id"]),
            query=str(item["query"]),
            predicted_tool=str(item["predicted_tool"]),
            retrieved_tools=[str(name) for name in item.get("retrieved_tools", [])],
            scale_candidate_pool=[str(name) for name in item.get("scale_candidate_pool", [])],
            correct_tools=[str(name) for name in item.get("correct_tools", [])],
            is_correct=item.get("is_correct"),
        )
        for item in _read_jsonl(path)
    ]


def _extract_tool_names_from_context(context: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"TOOL_NAME:\s*([^\n\r]+)", context)]


def _tools_digest(tools: list[ToolSpec], generated_queries: dict[str, list[str]] | None = None) -> str:
    import hashlib

    digest = hashlib.sha1()
    for tool in tools:
        digest.update(
            tool_to_rag_document(
                tool,
                generated_queries.get(tool.name, []) if generated_queries else [],
                include_generated=bool(generated_queries),
            ).encode("utf-8")
        )
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


def _sanitize_lightrag_content(content: str) -> str:
    """Prevent tiktoken from treating literal special-token strings as control tokens."""

    return re.sub(r"<\|([^|\r\n]{1,100})\|>", r"< |\1| >", content)


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
        content = tool_to_rag_document(
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
        return "previous build did not finish"
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


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)
