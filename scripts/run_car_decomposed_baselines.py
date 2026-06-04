from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_bfcl_baselines import make_dense_embedder, make_embedding_progress, method_setting  # noqa: E402
from scripts.run_bfcl_full_toollery_variants import ManualBM25Index  # noqa: E402
from scripts.run_car_baselines import (  # noqa: E402
    load_car_samples,
    load_car_tools,
    make_full_toollery_index,
    make_retriever,
)
from toollery.baselines import (  # noqa: E402
    BaselinePrediction,
    RetrievalResult,
    ensure_safe_output_path,
    estimate_prompt_tokens,
    load_generated_queries,
    summarize_predictions,
    write_json,
    write_jsonl,
)
from toollery.schemas import ManualEntry, ToolCandidate, ToolSpec  # noqa: E402


SPLIT_RE = re.compile(r"(?:并且|同时|以及|然后|还有|另外|再|和|与|及|，|,|、|；|;)")
ACTION_TERMS = (
    "取消",
    "关闭",
    "关掉",
    "禁用",
    "退出",
    "打开",
    "开启",
    "启动",
    "上锁",
    "锁上",
    "解锁",
    "开锁",
    "切换",
    "设置",
    "设为",
    "调到",
    "调高",
    "调低",
    "调大",
    "调小",
    "放大",
    "缩小",
    "增加",
    "减少",
    "升高",
    "降低",
)


def decompose_query(query: str, *, keep_original: bool = True) -> list[str]:
    compact = re.sub(r"\s+", "", query.strip())
    if not compact:
        return []
    parts = [part.strip() for part in SPLIT_RE.split(compact) if len(part.strip()) >= 2]
    if len(parts) <= 1:
        return [compact]

    leading_action = next((term for term in ACTION_TERMS if compact.startswith(term)), "")
    normalized: list[str] = []
    for part in parts:
        if (
            leading_action
            and not any(term in part for term in ACTION_TERMS)
            and not part.startswith(leading_action)
        ):
            part = leading_action + part
        if part not in normalized:
            normalized.append(part)
    if keep_original and compact not in normalized:
        normalized.append(compact)
    return normalized


def merge_ranked_lists(
    ranked_lists: list[list[tuple[str, float, str]]],
    tools: list[ToolSpec],
    top_k: int,
) -> list[str]:
    limit = min(top_k, len(tools))
    retrieved: list[str] = []
    seen: set[str] = set()
    max_len = max((len(items) for items in ranked_lists), default=0)
    for rank in range(max_len):
        for hits in ranked_lists:
            if rank >= len(hits):
                continue
            name = hits[rank][0]
            if name in seen:
                continue
            seen.add(name)
            retrieved.append(name)
            if len(retrieved) >= limit:
                return retrieved
    for tool in tools:
        if len(retrieved) >= limit:
            break
        if tool.name not in seen:
            seen.add(tool.name)
            retrieved.append(tool.name)
    return retrieved


def retrieval_results_to_hits(results: list[RetrievalResult]) -> list[tuple[str, float, str]]:
    return [(item.tool.name, item.score, item.document) for item in results]


def tool_candidates_to_hits(results: list[ToolCandidate]) -> list[tuple[str, float, str]]:
    hits: list[tuple[str, float, str]] = []
    for item in results:
        support = item.supporting_queries[0].query if item.supporting_queries else ""
        hits.append((item.tool.name, item.score, support))
    return hits


class ManualBM25ToolRetriever:
    def __init__(self, tools: list[ToolSpec], generated_queries: dict[str, list[str]]) -> None:
        tool_names = {tool.name for tool in tools}
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.manual = [
            ManualEntry(query=query, tool_name=tool_name, source="car_generated")
            for tool_name, queries in generated_queries.items()
            if tool_name in tool_names
            for query in queries
        ]
        if not self.manual:
            raise ValueError("No generated queries matched tools in the car tool pool")
        self.index = ManualBM25Index(self.manual)

    def retrieve(self, query: str, tools: list[ToolSpec], top_k: int) -> list[RetrievalResult]:
        allowed_names = {tool.name for tool in tools}
        grouped: dict[str, list[float]] = defaultdict(list)
        for idx, score in self.index.search(query, proxy_top_k=len(self.manual)):
            tool_name = self.manual[idx].tool_name
            if tool_name in allowed_names:
                grouped[tool_name].append(score)

        candidates: list[tuple[str, float]] = []
        for tool_name, scores in grouped.items():
            ranked = sorted(scores, reverse=True)
            score = sum(ranked[:3]) / min(len(ranked), 3)
            score += 0.05 * min(len(ranked), 5)
            candidates.append((tool_name, score))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalResult(tool=self.tools_by_name[name], score=score, document="")
            for name, score in candidates[:top_k]
            if name in self.tools_by_name
        ]


def run_decomposed_method(
    *,
    method_name: str,
    samples: list[Any],
    tools: list[ToolSpec],
    retrieve_hits: Callable[[str, list[ToolSpec]], list[tuple[str, float, str]]],
    top_k: int,
) -> list[dict[str, Any]]:
    tools_by_name = {tool.name: tool for tool in tools}
    predictions: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        start = time.perf_counter()
        retrieval_start = time.perf_counter()
        subqueries = decompose_query(sample.query)
        ranked_lists = [retrieve_hits(subquery, sample.tools) for subquery in subqueries]
        retrieved = merge_ranked_lists(ranked_lists, sample.tools, top_k)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

        retrieved_tools = [tools_by_name[name] for name in retrieved if name in tools_by_name]
        correct_tools = sample.correct_tools
        prediction = BaselinePrediction(
            method_setting="decomposed",
            method_name=method_name,
            benchmark="car_fullpool",
            sample_id=sample.sample_id,
            query=sample.query,
            candidate_pool_size=len(sample.tools),
            retrieved_candidates=retrieved,
            correct_candidates=correct_tools,
            top_k_hit=bool(set(retrieved) & set(correct_tools)) if correct_tools else None,
            final_prediction=retrieved[0] if retrieved else "",
            final_success=(retrieved[0] in correct_tools) if retrieved and correct_tools else None,
            prompt_tokens=estimate_prompt_tokens(sample.query, retrieved_tools),
            completion_tokens=None,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=0.0,
            llm_latency_ms=0.0,
            total_latency_ms=(time.perf_counter() - start) * 1000.0,
        )
        row = prediction.__dict__.copy()
        row["decomposed_queries"] = subqueries
        predictions.append(row)
        if index == len(samples) or index % 500 == 0:
            print(f"[{method_name}] evaluated {index}/{len(samples)} samples", file=sys.stderr, flush=True)
    return predictions


def prediction_objects(rows: list[dict[str, Any]]) -> list[BaselinePrediction]:
    fields = set(BaselinePrediction.__dataclass_fields__)
    return [BaselinePrediction(**{key: row[key] for key in fields}) for row in rows]


def make_retrieve_hits(method_name: str, args: argparse.Namespace, tools: list[ToolSpec], generated_queries: dict[str, list[str]]):
    if method_name == "decomposed_full_toollery_bm25":
        retriever = ManualBM25ToolRetriever(tools, generated_queries)
        return lambda query, pool: retrieval_results_to_hits(retriever.retrieve(query, pool, len(pool)))

    if method_name == "decomposed_full_toollery":
        index = make_full_toollery_index("full_toollery", tools, args, generated_queries)

        def retrieve(query: str, pool: list[ToolSpec]) -> list[tuple[str, float, str]]:
            return tool_candidates_to_hits(
                index.retrieve_tools(
                    query,
                    tool_top_k=len(index.tools_by_name),
                    proxy_top_k=len(index.manual),
                )
            )

        return retrieve

    base_method = method_name.removeprefix("decomposed_")
    retriever = make_retriever(base_method, args, generated_queries)
    return lambda query, pool: retrieval_results_to_hits(retriever.retrieve(query, pool, len(pool)))


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
    parser.add_argument("--dense-embedding-backend", default="local-hf")
    parser.add_argument("--dense-embedding-api-key")
    parser.add_argument("--dense-embedding-base-url")
    parser.add_argument("--dense-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dense-embedding-dim", type=int, default=1536)
    parser.add_argument("--dense-embedding-timeout", type=int)
    parser.add_argument("--dense-embedding-device")
    parser.add_argument("--dense-embedding-batch-size", type=int, default=16)
    parser.add_argument("--dense-embedding-max-length", type=int, default=256)
    parser.add_argument("--embedding-pooling", dest="dense_embedding_pooling", choices=("last", "mean", "cls"), default="last")
    parser.add_argument("--embedding-dtype", dest="dense_embedding_dtype")
    parser.add_argument("--embedding-trust-remote-code", dest="dense_embedding_trust_remote_code", action="store_true", default=False)
    parser.add_argument("--embedding-local-files-only", dest="dense_embedding_local_files_only", action="store_true", default=False)
    parser.add_argument("--force-rebuild-embeddings", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_safe_output_path(args.output_dir)
    tools = load_car_tools(args.tools)
    samples = load_car_samples(args.samples, tools, args.limit_samples)
    generated_queries = load_generated_queries(args.generated_queries)
    output_root = Path(args.output_dir)
    summary: dict[str, Any] = {}
    for method_name in [item.strip() for item in args.methods.split(",") if item.strip()]:
        retrieve_hits = make_retrieve_hits(method_name, args, tools, generated_queries)
        rows = run_decomposed_method(
            method_name=method_name,
            samples=samples,
            tools=tools,
            retrieve_hits=retrieve_hits,
            top_k=args.top_k,
        )
        method_dir = output_root / method_name
        write_jsonl(method_dir / "predictions.jsonl", rows)
        method_summary = {
            "method_setting": "decomposed",
            "base_method_setting": method_setting(method_name.removeprefix("decomposed_"))
            if not method_name.startswith("decomposed_full_toollery")
            else "full_toollery",
            "method_name": method_name,
            "benchmark": "car_fullpool",
            "tools": str(Path(args.tools).resolve()),
            "samples": str(Path(args.samples).resolve()),
            "generated_queries": str(Path(args.generated_queries).resolve()),
            "top_k": args.top_k,
            "summary": summarize_predictions(prediction_objects(rows)),
        }
        write_json(method_dir / "summary.json", method_summary)
        summary[method_name] = method_summary
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
