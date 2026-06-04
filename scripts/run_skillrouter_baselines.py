from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_skillrouter_toollery import (  # noqa: E402
    TIER_NAMES,
    _format_path,
    _include_task,
    _load_jsonl,
    _resolve_data_root,
    evaluate_predictions,
    load_skillrouter_pool,
)
from toollery.baselines import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    EmbeddingConfig,
    run_full_context_llm_baseline,
    ensure_safe_output_path,
    load_generated_queries,
    make_rag_retriever,
    write_json,
)
from toollery.embeddings import HashingTfidfEmbedder, LocalHFEmbedder, OpenAICompatibleEmbedder  # noqa: E402
from toollery.llm import HeuristicFinalSelector, OpenAICompatibleLLM  # noqa: E402


DEFAULT_METHODS = [
    "bm25_raw",
    "dense_raw",
    "raganything_raw",
    "bm25_query_augmented",
    "dense_query_augmented",
    "raganything_query_augmented",
    "full_context_llm_raw",
]


def main() -> None:
    args = parse_args()
    ensure_safe_output_path(args.output_dir)
    ensure_safe_output_path(args.dense_embedding_cache_dir)
    skillrouter_root = Path(args.skillrouter_root)
    data_root = _resolve_data_root(skillrouter_root, args.data_root)
    output_root = Path(args.output_dir)
    tasks = _load_jsonl(data_root / "tasks.jsonl")
    relevance = json.loads((data_root / "relevance.json").read_text(encoding="utf-8"))
    generated_queries = load_generated_queries(args.generated_queries)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    summary: dict[str, Any] = {}

    for tier in args.tiers:
        pool = load_skillrouter_pool(
            data_root / tier,
            limit=args.limit_pool,
            body_chars=args.skill_body_chars,
        )
        pool_ids = {tool.name for tool in pool}
        tier_summary: dict[str, Any] = {}
        for method_name in methods:
            if method_name == "full_context_llm_raw":
                predictions = run_full_context(
                    tasks=tasks,
                    relevance=relevance,
                    pool=pool,
                    selector=make_selector(args),
                    task_mode=args.task_mode,
                    limit_tasks=args.limit_tasks,
                )
                retriever = None
            else:
                retriever = make_retriever(method_name, args, generated_queries)
                predictions = run_retrieval(
                    tasks=tasks,
                    relevance=relevance,
                    pool=pool,
                    retriever=retriever,
                    top_k=args.top_k,
                    task_mode=args.task_mode,
                    limit_tasks=args.limit_tasks,
                )
            metrics = evaluate_predictions(
                tasks=tasks,
                relevance=relevance,
                predictions=predictions,
                pool_ids=pool_ids,
                task_mode=args.task_mode,
            )
            method_root = output_root / method_name
            predictions_path = _format_path(args.predictions_out, tier, method_root)
            metrics_path = _format_path(args.metrics_out, tier, method_root)
            write_json(predictions_path, predictions)
            payload = {
                "method_setting": method_setting(method_name),
                "method_name": method_name,
                "benchmark": "skillrouter",
                "tier": tier,
                "pool_size": len(pool),
                "top_k": args.top_k,
                "metrics": metrics,
            }
            write_json(metrics_path, payload)
            tier_summary[method_name] = payload
            if retriever is not None:
                close = getattr(retriever, "close", None)
                if close is not None:
                    close()
        summary[tier] = tier_summary

    summary_path = _format_path(args.summary_out, "summary", output_root)
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw and query-augmented SkillRouter baseline retrievers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skillrouter-root",
        type=str,
        required=True,
        metavar="DIR",
        help=(
            "Required string path to the SkillRouter repository or data root. "
            "The script resolves --data-root relative to this directory when needed."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/eval_core",
        metavar="DIR",
        help=(
            "String path to SkillRouter Eval Core data. "
            "Relative paths are resolved under --skillrouter-root."
        ),
    )
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        choices=sorted(TIER_NAMES),
        default=["easy", "hard"],
        metavar="TIER",
        help="One or more SkillRouter tiers to evaluate.",
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
        "--generated-queries",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Optional string path to a normalized generated-query artifact. "
            "Only methods ending in '_query_augmented' append these queries to searchable skill documents."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        metavar="N",
        help="Integer number of ranked skill candidates returned for each task.",
    )
    parser.add_argument(
        "--task-mode",
        type=str,
        choices=["core", "all", "single"],
        default="core",
        help=(
            "String task filter passed to the SkillRouter evaluator. "
            "'core' evaluates core tasks, 'all' keeps all relevant tasks, and 'single' keeps single-skill tasks."
        ),
    )
    parser.add_argument(
        "--limit-pool",
        type=int,
        default=None,
        metavar="N",
        help="Optional integer cap on the number of skills loaded from each tier pool; useful for smoke tests.",
    )
    parser.add_argument(
        "--limit-tasks",
        type=int,
        default=None,
        metavar="N",
        help="Optional integer cap on the number of tasks evaluated per tier; useful for smoke tests.",
    )
    parser.add_argument(
        "--skill-body-chars",
        type=int,
        default=8000,
        metavar="N",
        help=(
            "Integer maximum number of characters retained from each skill body/manual when building raw skill documents."
        ),
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
        default="outputs/experiments_v2/skillrouter/rag_cache",
        metavar="DIR",
        help="String directory path for RAG-Anything/LightRAG cache files, scoped per method under this root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments_v2/skillrouter/baselines",
        metavar="DIR",
        help="String output directory for per-method retrieval files, metrics files, and the aggregate summary.",
    )
    parser.add_argument(
        "--dense-embedding-cache-dir",
        type=str,
        default="outputs/experiments_v2/skillrouter/dense_embedding_cache",
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
        "--predictions-out",
        type=str,
        default="retrieval/{tier}.json",
        metavar="TEMPLATE",
        help=(
            "String filename template, relative to each method output directory, for task_id-to-ranked-skill predictions. "
            "Use '{tier}' as the tier placeholder."
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="metrics/{tier}.json",
        metavar="TEMPLATE",
        help=(
            "String filename template, relative to each method output directory, for SkillRouter-compatible metrics. "
            "Use '{tier}' as the tier placeholder."
        ),
    )
    parser.add_argument(
        "--summary-out",
        type=str,
        default="summary.json",
        metavar="PATH",
        help=(
            "String output filename or template for the aggregate run summary. "
            "It is resolved under --output-dir."
        ),
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


def run_retrieval(
    *,
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    pool: list[Any],
    retriever: Any,
    top_k: int,
    task_mode: str,
    limit_tasks: int | None,
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    included_tasks = [task for task in tasks if _include_task(str(task["task_id"]), relevance, task_mode)]
    if limit_tasks is not None:
        included_tasks = included_tasks[:limit_tasks]
    for task in included_tasks:
        task_id = str(task["task_id"])
        hits = retriever.retrieve(str(task["instruction_text"]), pool, min(top_k, len(pool)))
        predictions[task_id] = [hit.tool.name for hit in hits]
    return predictions


def run_full_context(
    *,
    tasks: list[dict[str, Any]],
    relevance: dict[str, Any],
    pool: list[Any],
    selector: Any,
    task_mode: str,
    limit_tasks: int | None,
) -> dict[str, list[str]]:
    predictions: dict[str, list[str]] = {}
    included_tasks = [task for task in tasks if _include_task(str(task["task_id"]), relevance, task_mode)]
    if limit_tasks is not None:
        included_tasks = included_tasks[:limit_tasks]
    for task in included_tasks:
        task_id = str(task["task_id"])
        rel_entry = relevance.get(task_id, {})
        gt_ids = rel_entry.get("core_gt_ids", rel_entry.get("gt_skill_ids", []))
        row = run_full_context_llm_baseline(
            query=str(task["instruction_text"]),
            tools=pool,
            correct_tools=[str(item) for item in gt_ids],
            selector=selector,
            method_name="full_context_llm_raw",
            benchmark="skillrouter",
            sample_id=task_id,
        )
        predictions[task_id] = [row.final_prediction] if row.final_prediction else []
    return predictions


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
