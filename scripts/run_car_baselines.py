from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bfcl_baselines import (  # noqa: E402
    make_dense_embedder,
    make_embedding_progress,
    method_setting,
    run_full_toollery_retrieval_baseline,
    run_global_filtered_retrieval_baseline,
)
from scripts.run_bfcl_full_toollery_variants import run_method as run_manual_method  # noqa: E402
from toollery.baselines import (  # noqa: E402
    BM25Retriever,
    BaselinePrediction,
    DenseRetriever,
    EmbeddingConfig,
    RetrievalResult,
    build_tool_document,
    ensure_safe_output_path,
    estimate_prompt_tokens,
    load_generated_queries,
    make_rag_retriever,
    run_retrieval_baseline,
    summarize_predictions,
    write_json,
    write_jsonl,
)
from toollery.embeddings import cosine  # noqa: E402
from toollery.retrieval import ProxyQueryIndex  # noqa: E402
from toollery.schemas import ManualEntry, ToolSpec  # noqa: E402
from toollery.text import tokenize  # noqa: E402


@dataclass(frozen=True)
class CarSample:
    sample_id: str
    query: str
    correct_tools: list[str]
    tools: list[ToolSpec]


def load_car_tools(path: str | Path) -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tools: list[ToolSpec] = []
    for name, meta in data.items():
        parameters = {"type": "object", "properties": {}, "required": []}
        for item in meta.get("inputs", []) or []:
            param_name = str(item.get("name", "")).strip()
            if not param_name:
                continue
            param_type = str(item.get("type", "string")).strip().lower() or "string"
            if param_type not in {"string", "number", "integer", "boolean", "array", "object"}:
                param_type = "string"
            parameters["properties"][param_name] = {
                "type": param_type,
                "description": str(item.get("desc", "")).strip(),
            }
        tools.append(
            ToolSpec(
                name=str(meta.get("actionName") or name),
                description=str(meta.get("description") or meta.get("actionNameZh") or ""),
                parameters=parameters,
                category=str(meta.get("actionClz") or "car_intent"),
            )
        )
    return tools


def load_car_samples(path: str | Path, tools: list[ToolSpec], limit: int | None = None) -> list[CarSample]:
    samples: list[CarSample] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        samples.append(
            CarSample(
                sample_id=str(item["sample_id"]),
                query=str(item["query"]),
                correct_tools=[str(name) for name in item["correct_tools"]],
                tools=tools,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def make_retriever(method_name: str, args: argparse.Namespace, generated_queries: dict[str, list[str]]):
    if method_name.endswith("_query_augmented"):
        if method_name.startswith("bm25_"):
            return QueryAugmentedBM25Retriever(generated_queries=generated_queries)
        if method_name.startswith("dense_"):
            return QueryAugmentedDenseRetriever(
                embedder=make_dense_embedder(args),
                generated_queries=generated_queries,
                embedding_cache_path=Path(args.dense_embedding_cache_dir) / method_name,
                force_rebuild_embeddings=args.force_rebuild_embeddings,
                embedding_progress_callback=make_embedding_progress(method_name),
            )
        if method_name.startswith("raganything_"):
            return QueryAugmentedDenseRetriever(
                embedder=make_dense_embedder(args),
                generated_queries=generated_queries,
                embedding_cache_path=Path(args.dense_embedding_cache_dir) / method_name,
                force_rebuild_embeddings=args.force_rebuild_embeddings,
                embedding_progress_callback=make_embedding_progress(method_name),
            )
    include_generated = False
    if method_name.startswith("bm25_"):
        return BM25Retriever(generated_queries=generated_queries, include_generated=include_generated)
    if method_name.startswith("dense_"):
        return DenseRetriever(
            embedder=make_dense_embedder(args),
            generated_queries=generated_queries,
            include_generated=include_generated,
            embedding_cache_path=Path(args.dense_embedding_cache_dir) / method_name,
            force_rebuild_embeddings=args.force_rebuild_embeddings,
            embedding_progress_callback=make_embedding_progress(method_name),
        )
    if method_name.startswith("raganything_"):
        return make_rag_retriever(
            args.backend,
            Path(args.rag_working_dir) / method_name,
            embedding_config=EmbeddingConfig(
                backend=args.embedding_backend,
                api_key=args.embedding_api_key,
                base_url=args.embedding_base_url,
                model=args.embedding_model,
                dim=args.embedding_dim,
                timeout=args.embedding_timeout,
                device=args.embedding_device,
                batch_size=args.embedding_batch_size,
                max_length=args.embedding_max_length,
                pooling=args.embedding_pooling,
                dtype=args.embedding_dtype,
                trust_remote_code=args.embedding_trust_remote_code,
                local_files_only=args.embedding_local_files_only,
            ),
            generated_queries=generated_queries,
            include_generated=include_generated,
        )
    raise ValueError(f"unknown method: {method_name}")


@dataclass(frozen=True)
class QueryAugmentedUnit:
    tool: ToolSpec
    document: str


class QueryAugmentedBM25Retriever:
    def __init__(self, generated_queries: dict[str, list[str]]) -> None:
        self.generated_queries = generated_queries
        self._cache: dict[str, tuple[list[QueryAugmentedUnit], list[list[str]], dict[str, int], float]] = {}

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        key = ",".join(sorted(tool.name for tool in tools))
        cached = self._cache.get(key)
        if cached is None:
            units = make_query_augmented_units(tools, self.generated_queries)
            tokenized = [tokenize(unit.document) for unit in units]
            dfs: Counter[str] = Counter()
            for tokens in tokenized:
                dfs.update(set(tokens))
            avgdl = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
            cached = (units, tokenized, dict(dfs), avgdl)
            self._cache[key] = cached
        units, tokenized, dfs, avgdl = cached
        doc_count = max(len(units), 1)
        query_terms = tokenize(query)
        grouped: dict[str, list[tuple[float, QueryAugmentedUnit]]] = defaultdict(list)
        for unit, tokens in zip(units, tokenized):
            tf = Counter(tokens)
            score = 0.0
            dl = len(tokens)
            for term in query_terms:
                if term not in tf:
                    continue
                df = dfs.get(term, 0)
                idf = math.log(1.0 + (doc_count - df + 0.5) / (df + 0.5))
                denom = tf[term] + 1.5 * (1.0 - 0.75 + 0.75 * dl / max(avgdl, 1e-9))
                score += idf * tf[term] * 2.5 / denom
            grouped[unit.tool.name].append((score, unit))
        return aggregate_unit_hits(grouped, tools, top_k)


class QueryAugmentedDenseRetriever:
    def __init__(
        self,
        *,
        embedder: Any,
        generated_queries: dict[str, list[str]],
        embedding_cache_path: str | Path | None = None,
        force_rebuild_embeddings: bool = False,
        embedding_progress_callback: object | None = None,
    ) -> None:
        self.embedder = embedder
        self.generated_queries = generated_queries
        self.embedding_cache_path = Path(embedding_cache_path) if embedding_cache_path else None
        self.force_rebuild_embeddings = force_rebuild_embeddings
        self.embedding_progress_callback = embedding_progress_callback
        self._cache: dict[str, tuple[list[QueryAugmentedUnit], list[list[float]]]] = {}

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        key = ",".join(sorted(tool.name for tool in tools))
        cached = self._cache.get(key)
        if cached is None:
            units = make_query_augmented_units(tools, self.generated_queries)
            documents = [unit.document for unit in units]
            self.embedder.fit(documents)
            vectors = load_or_build_unit_vectors(
                documents,
                self.embedder,
                self.embedding_cache_path,
                self.force_rebuild_embeddings,
                self.embedding_progress_callback,
            )
            cached = (units, vectors)
            self._cache[key] = cached
        units, vectors = cached
        query_vector = self.embedder.encode(query)
        grouped: dict[str, list[tuple[float, QueryAugmentedUnit]]] = defaultdict(list)
        for unit, vector in zip(units, vectors):
            grouped[unit.tool.name].append((cosine(query_vector, vector), unit))
        return aggregate_unit_hits(grouped, tools, top_k)


def make_query_augmented_units(
    tools: list[ToolSpec],
    generated_queries: dict[str, list[str]],
) -> list[QueryAugmentedUnit]:
    units: list[QueryAugmentedUnit] = []
    for tool in tools:
        queries = [query for query in generated_queries.get(tool.name, []) if query]
        if not queries:
            units.append(QueryAugmentedUnit(tool=tool, document=build_tool_document(tool)))
            continue
        metadata = build_tool_document(tool)
        for query in queries:
            units.append(
                QueryAugmentedUnit(
                    tool=tool,
                    document=f"{metadata}\nGENERATED_QUERY: {query}",
                )
            )
    return units


def aggregate_unit_hits(
    grouped: dict[str, list[tuple[float, QueryAugmentedUnit]]],
    tools: list[ToolSpec],
    top_k: int,
) -> list[RetrievalResult]:
    tools_by_name = {tool.name: tool for tool in tools}
    candidates: list[tuple[str, float, str]] = []
    for tool_name, scored_units in grouped.items():
        ranked = sorted(scored_units, key=lambda item: item[0], reverse=True)
        top_scores = [score for score, _ in ranked[:3]]
        score = sum(top_scores) / max(len(top_scores), 1)
        score += 0.05 * min(len(ranked), 5)
        candidates.append((tool_name, score, ranked[0][1].document))
    candidates.sort(key=lambda item: item[1], reverse=True)
    hits = [
        RetrievalResult(tool=tools_by_name[tool_name], score=score, document=document)
        for tool_name, score, document in candidates[:top_k]
        if tool_name in tools_by_name
    ]
    return hits


def load_or_build_unit_vectors(
    documents: list[str],
    embedder: Any,
    cache_path: Path | None,
    force_rebuild: bool,
    progress_callback: object | None,
) -> list[list[float]]:
    if cache_path is None:
        return embedder.encode_many(documents, progress_callback=progress_callback)
    cache_file = cache_path / "query_augmented_units.jsonl"
    if cache_file.exists() and not force_rebuild:
        vectors_by_doc: dict[str, list[float]] = {}
        with cache_file.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                vectors_by_doc[str(item["document"])] = [float(value) for value in item["vector"]]
        if all(document in vectors_by_doc for document in documents):
            if progress_callback:
                progress_callback(len(documents), len(documents), "cache_hit")
            return [vectors_by_doc[document] for document in documents]
    cache_path.mkdir(parents=True, exist_ok=True)
    vectors = embedder.encode_many(documents, progress_callback=progress_callback)
    tmp = cache_file.with_suffix(".partial.jsonl")
    with tmp.open("w", encoding="utf-8") as handle:
        for document, vector in zip(documents, vectors):
            handle.write(json.dumps({"document": document, "vector": vector}, ensure_ascii=False) + "\n")
    tmp.replace(cache_file)
    return vectors


def make_full_toollery_index(
    method_name: str,
    tools: list[ToolSpec],
    args: argparse.Namespace,
    generated_queries: dict[str, list[str]],
) -> ProxyQueryIndex:
    if not generated_queries:
        raise ValueError("--generated-queries is required when running full_toollery")
    tool_names = {tool.name for tool in tools}
    manual = [
        ManualEntry(query=query, tool_name=tool_name, source="car_generated")
        for tool_name, queries in generated_queries.items()
        if tool_name in tool_names
        for query in queries
    ]
    if not manual:
        raise ValueError("No generated queries matched tools in the car tool pool")
    return ProxyQueryIndex(
        tools,
        manual,
        embedder=make_dense_embedder(args),
        embedding_cache_path=Path(args.dense_embedding_cache_dir) / method_name,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
        embedding_progress_callback=make_embedding_progress(method_name),
    )


def run_global_method(
    *,
    method_name: str,
    samples: list[CarSample],
    tools: list[ToolSpec],
    args: argparse.Namespace,
    generated_queries: dict[str, list[str]],
) -> list[BaselinePrediction]:
    retriever = make_retriever(method_name, args, generated_queries)
    predictions: list[BaselinePrediction] = []
    for index, sample in enumerate(samples, start=1):
        predictions.append(
            run_global_filtered_retrieval_baseline(
                query=sample.query,
                search_tools=tools,
                tools=sample.tools,
                correct_tools=sample.correct_tools,
                retriever=retriever,
                method_setting=method_setting(method_name),
                method_name=method_name,
                benchmark="car_fullpool",
                sample_id=sample.sample_id,
                top_k=args.top_k,
            )
        )
        if index == len(samples) or index % 500 == 0:
            print(f"[{method_name}] evaluated {index}/{len(samples)} samples", file=sys.stderr, flush=True)
    close = getattr(retriever, "close", None)
    if close is not None:
        close()
    return predictions


def run_full_toollery_dense(
    *,
    samples: list[CarSample],
    tools: list[ToolSpec],
    args: argparse.Namespace,
    generated_queries: dict[str, list[str]],
) -> list[BaselinePrediction]:
    index = make_full_toollery_index("full_toollery", tools, args, generated_queries)
    predictions: list[BaselinePrediction] = []
    for row_idx, sample in enumerate(samples, start=1):
        predictions.append(
            run_full_toollery_retrieval_baseline(
                query=sample.query,
                tools=sample.tools,
                correct_tools=sample.correct_tools,
                index=index,
                sample_id=sample.sample_id,
                top_k=args.top_k,
            )
        )
        if row_idx == len(samples) or row_idx % 500 == 0:
            print(f"[full_toollery] evaluated {row_idx}/{len(samples)} samples", file=sys.stderr, flush=True)
    return predictions


def run_full_toollery_manual(
    *,
    method_name: str,
    samples: list[CarSample],
    tools: list[ToolSpec],
    args: argparse.Namespace,
    generated_queries: dict[str, list[str]],
) -> list[BaselinePrediction]:
    if method_name == "full_toollery_bm25":
        manual_method = "bm25"
    elif method_name == "full_toollery_rag":
        manual_method = "rag"
    else:
        raise ValueError(f"unknown full-toollery manual method: {method_name}")
    tool_names = {tool.name for tool in tools}
    manual = [
        ManualEntry(query=query, tool_name=tool_name, source="car_generated")
        for tool_name, queries in generated_queries.items()
        if tool_name in tool_names
        for query in queries
    ]
    return run_manual_method(method=manual_method, samples=samples, manual=manual, top_k=args.top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--generated-queries", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument("--backend", choices=("raganything", "tfidf", "lightrag"), default="raganything")
    parser.add_argument("--rag-working-dir", default="outputs/experiments_v2/car/rag_cache")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dense-embedding-cache-dir", default="outputs/experiments_v2/car/dense_embedding_cache")
    parser.add_argument("--force-rebuild-embeddings", action="store_true", default=False)
    parser.add_argument("--dense-embedding-backend", choices=("tfidf", "openai-compatible", "local-hf"), default="tfidf")
    parser.add_argument("--dense-embedding-model")
    parser.add_argument("--dense-embedding-device")
    parser.add_argument("--dense-embedding-batch-size", type=int, default=8)
    parser.add_argument("--dense-embedding-max-length", type=int, default=512)
    parser.add_argument("--dense-embedding-pooling", choices=("last", "mean", "cls"), default="last")
    parser.add_argument("--dense-embedding-dtype")
    parser.add_argument("--dense-embedding-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--dense-embedding-local-files-only", action="store_true", default=False)
    parser.add_argument("--embedding-backend", choices=("openai-compatible", "local-hf"), default="openai-compatible")
    parser.add_argument("--embedding-api-key")
    parser.add_argument("--embedding-base-url")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-dim", type=int, default=1536)
    parser.add_argument("--embedding-timeout", type=int)
    parser.add_argument("--embedding-device")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-max-length", type=int, default=512)
    parser.add_argument("--embedding-pooling", choices=("last", "mean", "cls"), default="last")
    parser.add_argument("--embedding-dtype")
    parser.add_argument("--embedding-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--embedding-local-files-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_safe_output_path(args.output_dir)
    ensure_safe_output_path(args.dense_embedding_cache_dir)
    tools = load_car_tools(args.tools)
    samples = load_car_samples(args.samples, tools, args.limit_samples)
    generated_queries = load_generated_queries(args.generated_queries)
    output_root = Path(args.output_dir)
    summary: dict[str, Any] = {}
    for method_name in [item.strip() for item in args.methods.split(",") if item.strip()]:
        if method_name == "full_toollery":
            predictions = run_full_toollery_dense(
                samples=samples,
                tools=tools,
                args=args,
                generated_queries=generated_queries,
            )
        elif method_name in {"full_toollery_bm25", "full_toollery_rag"}:
            predictions = run_full_toollery_manual(
                method_name=method_name,
                samples=samples,
                tools=tools,
                args=args,
                generated_queries=generated_queries,
            )
        else:
            predictions = run_global_method(
                method_name=method_name,
                samples=samples,
                tools=tools,
                args=args,
                generated_queries=generated_queries,
            )
        method_dir = output_root / method_name
        write_jsonl(method_dir / "predictions.jsonl", (row.__dict__ for row in predictions))
        method_summary = {
            "method_setting": "full_toollery" if method_name.startswith("full_toollery") else method_setting(method_name),
            "method_name": method_name,
            "benchmark": "car_fullpool",
            "tools": str(Path(args.tools).resolve()),
            "samples": str(Path(args.samples).resolve()),
            "generated_queries": str(Path(args.generated_queries).resolve()),
            "top_k": args.top_k,
            "summary": summarize_predictions(predictions),
        }
        write_json(method_dir / "summary.json", method_summary)
        summary[method_name] = method_summary
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
