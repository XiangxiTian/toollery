from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toollery.baselines import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    EmbeddingConfig,
    BaselinePrediction,
    ensure_safe_output_path,
    estimate_prompt_tokens,
    load_generated_queries,
    make_rag_retriever,
    run_full_context_llm_baseline,
    run_retrieval_baseline,
    summarize_predictions,
    write_json,
    write_jsonl,
)
from toollery.bfcl import load_bfcl_scaled_dataset  # noqa: E402
from toollery.embeddings import HashingTfidfEmbedder, LocalHFEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from toollery.llm import HeuristicFinalSelector, OpenAICompatibleLLM  # noqa: E402
from toollery.retrieval import ProxyQueryIndex  # noqa: E402
from toollery.schemas import ManualEntry  # noqa: E402


DEFAULT_METHODS = [
    "bm25_raw",
    "dense_raw",
    "raganything_raw",
    "bm25_query_augmented",
    "dense_query_augmented",
    "raganything_query_augmented",
    "full_toollery",
    "full_context_llm_raw",
]


def main() -> None:
    args = parse_args()
    ensure_safe_output_path(args.output_dir)
    ensure_safe_output_path(args.dense_embedding_cache_dir)
    generated_queries = load_generated_queries(args.generated_queries)
    samples = load_bfcl_scaled_dataset(args.scaled_data)
    if args.limit_samples is not None:
        samples = samples[: args.limit_samples]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    output_root = Path(args.output_dir)
    summary: dict[str, Any] = {}

    for method_name in methods:
        retriever = (
            None
            if method_name in {"full_context_llm_raw", "full_toollery"}
            else make_retriever(method_name, args, generated_queries)
        )
        selector = make_selector(args) if method_name == "full_context_llm_raw" else None
        full_toollery_index = (
            make_full_toollery_index(method_name, samples, args, generated_queries)
            if method_name == "full_toollery"
            else None
        )
        global_filter_tools = (
            _unique_tools(samples)
            if _uses_global_filtered_retrieval(method_name, args)
            else None
        )
        predictions: list[BaselinePrediction] = []
        for index, sample in enumerate(samples, start=1):
            if method_name == "full_context_llm_raw":
                predictions.append(
                    run_full_context_llm_baseline(
                        query=sample.query,
                        tools=sample.tools,
                        correct_tools=sample.correct_tools,
                        selector=selector,
                        method_name=method_name,
                        benchmark="bfcl_scaled",
                        sample_id=sample.sample_id,
                    )
                )
            elif full_toollery_index is not None:
                predictions.append(
                    run_full_toollery_retrieval_baseline(
                        query=sample.query,
                        tools=sample.tools,
                        correct_tools=sample.correct_tools,
                        index=full_toollery_index,
                        sample_id=sample.sample_id,
                        top_k=args.top_k,
                    )
                )
            elif global_filter_tools is not None:
                predictions.append(
                    run_global_filtered_retrieval_baseline(
                        query=sample.query,
                        search_tools=global_filter_tools,
                        tools=sample.tools,
                        correct_tools=sample.correct_tools,
                        retriever=retriever,
                        method_setting=method_setting(method_name),
                        method_name=method_name,
                        benchmark="bfcl_scaled",
                        sample_id=sample.sample_id,
                        top_k=args.top_k,
                    )
                )
            else:
                predictions.append(
                    run_retrieval_baseline(
                    query=sample.query,
                    tools=sample.tools,
                    correct_tools=sample.correct_tools,
                    retriever=retriever,
                    method_setting=method_setting(method_name),
                    method_name=method_name,
                    benchmark="bfcl_scaled",
                    sample_id=sample.sample_id,
                    top_k=args.top_k,
                    selector=None,
                    )
                )
            if index == len(samples) or index % 50 == 0:
                print(
                    f"[{method_name}] evaluated {index}/{len(samples)} samples",
                    file=sys.stderr,
                    flush=True,
                )
        method_dir = output_root / method_name
        write_jsonl(method_dir / "predictions.jsonl", (row.__dict__ for row in predictions))
        method_summary = {
            "method_setting": method_setting(method_name),
            "method_name": method_name,
            "benchmark": "bfcl_scaled",
            "scaled_data": str(Path(args.scaled_data).resolve()),
            "top_k": args.top_k,
            "summary": summarize_predictions(predictions),
        }
        write_json(method_dir / "summary.json", method_summary)
        summary[method_name] = method_summary
        if retriever is not None:
            close = getattr(retriever, "close", None)
            if close is not None:
                close()

    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _unique_tools(samples: list[Any]) -> list[Any]:
    tools_by_name: dict[str, Any] = {}
    for sample in samples:
        for tool in sample.tools:
            tools_by_name.setdefault(tool.name, tool)
    return list(tools_by_name.values())


def _uses_global_filtered_retrieval(method_name: str, args: argparse.Namespace) -> bool:
    if method_name.startswith("raganything_") and args.backend in {"raganything", "lightrag"}:
        return True
    return method_name.startswith("dense_") and args.dense_embedding_backend != "tfidf"


def make_full_toollery_index(
    method_name: str,
    samples: list[Any],
    args: argparse.Namespace,
    generated_queries: dict[str, list[str]],
) -> ProxyQueryIndex:
    if not generated_queries:
        raise ValueError("--generated-queries is required when running full_toollery")
    tools = _unique_tools(samples)
    tool_names = {tool.name for tool in tools}
    manual = [
        ManualEntry(query=query, tool_name=tool_name, source="bfcl_generated")
        for tool_name, queries in generated_queries.items()
        if tool_name in tool_names
        for query in queries
    ]
    if not manual:
        raise ValueError("No generated queries matched tools in the scaled dataset")
    return ProxyQueryIndex(
        tools,
        manual,
        embedder=make_dense_embedder(args),
        embedding_cache_path=Path(args.dense_embedding_cache_dir) / method_name,
        force_rebuild_embeddings=args.force_rebuild_embeddings,
        embedding_progress_callback=make_embedding_progress(method_name),
    )


def run_full_toollery_retrieval_baseline(
    *,
    query: str,
    tools: list[Any],
    correct_tools: list[str],
    index: ProxyQueryIndex,
    sample_id: str,
    top_k: int,
) -> BaselinePrediction:
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    allowed_names = {tool.name for tool in tools}
    tools_by_name = {tool.name: tool for tool in tools}
    retrieved: list[str] = []
    for candidate in index.retrieve_tools(
        query,
        tool_top_k=len(index.tools_by_name),
        proxy_top_k=len(index.manual),
    ):
        if candidate.tool.name in allowed_names and candidate.tool.name not in retrieved:
            retrieved.append(candidate.tool.name)
        if len(retrieved) >= min(top_k, len(tools)):
            break
    for tool in tools:
        if len(retrieved) >= min(top_k, len(tools)):
            break
        if tool.name not in retrieved:
            retrieved.append(tool.name)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
    retrieved_tools = [tools_by_name[name] for name in retrieved if name in tools_by_name]
    top_k_hit = bool(set(retrieved) & set(correct_tools)) if correct_tools else None
    final_prediction = retrieved[0] if retrieved else (tools[0].name if tools else "")
    final_success = final_prediction in correct_tools if correct_tools else None
    return BaselinePrediction(
        method_setting="full_toollery",
        method_name="full_toollery",
        benchmark="bfcl_scaled",
        sample_id=sample_id,
        query=query,
        candidate_pool_size=len(tools),
        retrieved_candidates=retrieved,
        correct_candidates=correct_tools,
        top_k_hit=top_k_hit,
        final_prediction=final_prediction,
        final_success=final_success,
        prompt_tokens=estimate_prompt_tokens(query, retrieved_tools),
        completion_tokens=None,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=0.0,
        llm_latency_ms=0.0,
        total_latency_ms=(time.perf_counter() - start) * 1000.0,
    )


def run_global_filtered_retrieval_baseline(
    *,
    query: str,
    search_tools: list[Any],
    tools: list[Any],
    correct_tools: list[str],
    retriever: Any,
    method_setting: str,
    method_name: str,
    benchmark: str,
    sample_id: str,
    top_k: int,
) -> BaselinePrediction:
    start = time.perf_counter()
    retrieval_start = time.perf_counter()
    search_hits = retriever.retrieve(query, search_tools, len(search_tools))
    allowed_names = {tool.name for tool in tools}
    tools_by_name = {tool.name: tool for tool in tools}
    retrieved: list[str] = []
    for hit in search_hits:
        if hit.tool.name in allowed_names and hit.tool.name not in retrieved:
            retrieved.append(hit.tool.name)
        if len(retrieved) >= min(top_k, len(tools)):
            break
    for tool in tools:
        if len(retrieved) >= min(top_k, len(tools)):
            break
        if tool.name not in retrieved:
            retrieved.append(tool.name)
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0
    retrieved_tools = [tools_by_name[name] for name in retrieved if name in tools_by_name]
    top_k_hit = bool(set(retrieved) & set(correct_tools)) if correct_tools else None
    final_prediction = retrieved[0] if retrieved else (tools[0].name if tools else "")
    final_success = final_prediction in correct_tools if correct_tools else None
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
        prompt_tokens=estimate_prompt_tokens(query, retrieved_tools),
        completion_tokens=None,
        retrieval_latency_ms=retrieval_latency_ms,
        rerank_latency_ms=0.0,
        llm_latency_ms=0.0,
        total_latency_ms=(time.perf_counter() - start) * 1000.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw and query-augmented BFCL baseline retrievers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scaled-data",
        type=str,
        required=True,
        metavar="PATH",
        help=(
            "Required string path to the BFCL scaled candidate-pool JSONL file. "
            "Each row should contain the user query, candidate tools, and gold tool names."
        ),
    )
    parser.add_argument(
        "--generated-queries",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional string path to a normalized generated-query artifact. "
            "Only methods ending in '_query_augmented' append these queries to searchable documents."
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_METHODS),
        metavar="CSV",
        help=(
            "Comma-separated string of methods to run. Supported values include "
            "bm25_raw, dense_raw, raganything_raw, bm25_query_augmented, "
            "dense_query_augmented, raganything_query_augmented, and full_context_llm_raw."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        metavar="N",
        help="Integer number of candidates retrieved per sample before reporting top-k hit or final selection.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        metavar="N",
        help="Optional integer cap on the number of BFCL samples to evaluate; useful for smoke tests.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=("raganything", "tfidf", "lightrag"),
        default="raganything",
        help=(
            "String backend used only by raganything_* methods. "
            "'raganything' and 'lightrag' use LightRAG storage; 'tfidf' is a local dependency-light smoke-test backend."
        ),
    )
    parser.add_argument(
        "--rag-working-dir",
        type=str,
        default="outputs/experiments_v2/bfcl/rag_cache",
        metavar="DIR",
        help="String directory path for RAG-Anything/LightRAG cache files, scoped per method under this root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments_v2/bfcl/baselines",
        metavar="DIR",
        help="String output directory for per-method predictions.jsonl files and summary.json files.",
    )
    parser.add_argument(
        "--dense-embedding-cache-dir",
        type=str,
        default="outputs/experiments_v2/bfcl/dense_embedding_cache",
        metavar="DIR",
        help=(
            "String directory for dense baseline document embedding caches. "
            "Each dense_* method writes a reusable JSONL cache with a .partial.jsonl file for interrupted runs."
        ),
    )
    parser.add_argument(
        "--force-rebuild-embeddings",
        action="store_true",
        default=False,
        help="Boolean flag that ignores existing dense embedding caches and rebuilds them from scratch.",
    )
    parser.add_argument(
        "--dense-embedding-backend",
        type=str,
        choices=("tfidf", "openai-compatible", "local-hf"),
        default="tfidf",
        help="String backend for dense_* methods: local hashing TF-IDF, remote OpenAI-compatible API, or local HuggingFace model.",
    )
    parser.add_argument(
        "--dense-embedding-model",
        type=str,
        default=None,
        metavar="MODEL_OR_PATH",
        help="Optional model name or local path for dense_* embeddings. For local-hf, point this to a downloaded model directory such as Qwen3-Embedding-4B.",
    )
    parser.add_argument(
        "--dense-embedding-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Optional device for local-hf dense embeddings, e.g. cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--dense-embedding-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Integer batch size for local-hf dense embedding encoding.",
    )
    parser.add_argument(
        "--dense-embedding-max-length",
        type=int,
        default=512,
        metavar="N",
        help="Integer max token length for local-hf dense embedding encoding.",
    )
    parser.add_argument(
        "--dense-embedding-pooling",
        type=str,
        choices=("last", "mean", "cls"),
        default="last",
        help="Pooling strategy for local-hf dense embeddings when using the Transformers fallback.",
    )
    parser.add_argument(
        "--dense-embedding-dtype",
        type=str,
        default=None,
        metavar="DTYPE",
        help="Optional dtype for local-hf model loading, e.g. float16, bfloat16, float32, or auto.",
    )
    parser.add_argument(
        "--dense-embedding-trust-remote-code",
        action="store_true",
        default=False,
        help="Boolean flag passed to HuggingFace loaders for local-hf dense embeddings.",
    )
    parser.add_argument(
        "--dense-embedding-local-files-only",
        action="store_true",
        default=False,
        help="Boolean flag that prevents HuggingFace from attempting downloads for local-hf dense embeddings.",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        choices=("openai-compatible", "local-hf"),
        default="openai-compatible",
        help=(
            "String embedding backend used only by raganything_* methods. "
            "Use openai-compatible for remote embedding APIs or local-hf for local HuggingFace models."
        ),
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="Optional string API key for the OpenAI-compatible embedding endpoint used by LightRAG backends.",
    )
    parser.add_argument(
        "--embedding-base-url",
        type=str,
        default=None,
        metavar="URL",
        help="Optional string base URL for an OpenAI-compatible embedding endpoint used by LightRAG backends.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        metavar="MODEL",
        help="String embedding model name or local HuggingFace model/path used by raganything_* methods.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1536,
        metavar="N",
        help="Integer embedding dimension requested from the OpenAI-compatible endpoint; local-hf infers this automatically.",
    )
    parser.add_argument(
        "--embedding-timeout",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Optional integer timeout in seconds for embedding requests made by LightRAG backends.",
    )
    parser.add_argument(
        "--embedding-device",
        type=str,
        default=None,
        metavar="DEVICE",
        help="Optional device for raganything_* local-hf embeddings, e.g. cpu, cuda, cuda:0, or mps.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        metavar="N",
        help="Integer batch size for raganything_* local-hf embedding encoding.",
    )
    parser.add_argument(
        "--embedding-max-length",
        type=int,
        default=512,
        metavar="N",
        help="Integer max token length for raganything_* local-hf embedding encoding.",
    )
    parser.add_argument(
        "--embedding-pooling",
        type=str,
        choices=("last", "mean", "cls"),
        default="last",
        help="Pooling strategy for raganything_* local-hf embeddings when using the Transformers fallback.",
    )
    parser.add_argument(
        "--embedding-dtype",
        type=str,
        default=None,
        metavar="DTYPE",
        help="Optional dtype for raganything_* local-hf model loading, e.g. float16, bfloat16, float32, or auto.",
    )
    parser.add_argument(
        "--embedding-trust-remote-code",
        action="store_true",
        default=False,
        help="Boolean flag passed to HuggingFace loaders for raganything_* local-hf embeddings.",
    )
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        default=False,
        help="Boolean flag that prevents HuggingFace downloads for raganything_* local-hf embeddings.",
    )
    parser.add_argument(
        "--selector",
        type=str,
        choices=("heuristic", "llm"),
        default="heuristic",
        help=(
            "String final selector used only by full_context_llm_raw. "
            "'heuristic' avoids model calls; 'llm' uses the configured OpenAI-compatible LLM client."
        ),
    )
    return parser.parse_args()


def make_retriever(method_name: str, args: argparse.Namespace, generated_queries: dict[str, list[str]]):
    include_generated = method_name.endswith("_query_augmented")
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


def method_setting(method_name: str) -> str:
    if method_name == "full_toollery":
        return "full_toollery"
    return "query_augmented_baseline" if method_name.endswith("_query_augmented") else "raw_baseline"


def make_selector(args: argparse.Namespace):
    if args.selector == "llm":
        return OpenAICompatibleLLM()
    return HeuristicFinalSelector()


def make_dense_embedder(args: argparse.Namespace):
    if args.dense_embedding_backend == "openai-compatible":
        return OpenAICompatibleEmbedder(
            model=args.dense_embedding_model or args.embedding_model,
            api_key=args.embedding_api_key,
            base_url=args.embedding_base_url,
            dimensions=args.embedding_dim,
            timeout=args.embedding_timeout or 120,
        )
    if args.dense_embedding_backend == "local-hf":
        return LocalHFEmbedder(
            model=args.dense_embedding_model,
            device=args.dense_embedding_device,
            batch_size=args.dense_embedding_batch_size,
            max_length=args.dense_embedding_max_length,
            pooling=args.dense_embedding_pooling,
            dtype=args.dense_embedding_dtype,
            trust_remote_code=args.dense_embedding_trust_remote_code,
            local_files_only=args.dense_embedding_local_files_only,
        )
    return HashingTfidfEmbedder()


def make_embedding_progress(method_name: str):
    last_done = -1
    printed_final = False
    printed_cache_hits: set[int] = set()

    def callback(done: int, total: int, status: str = "building") -> None:
        nonlocal last_done, printed_final
        if total <= 0:
            return
        if status == "cache_hit":
            if total in printed_cache_hits:
                return
            printed_cache_hits.add(total)
            print(f"[{method_name}] dense document embeddings: reusing cache {done}/{total}", file=sys.stderr, flush=True)
            return
        if done == last_done and status != "resumed":
            return
        last_done = done
        width = 28
        ratio = min(max(done / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        label = "dense document embeddings"
        suffix = " | resumed" if status == "resumed" else ""
        line = f"[{method_name}] {label}: [{bar}] {done}/{total} {ratio * 100:5.1f}%{suffix}"
        final = done >= total
        if final and printed_final:
            return
        printed_final = final
        print(line, file=sys.stderr, end="\n" if final else "\r", flush=True)

    return callback


if __name__ == "__main__":
    main()
