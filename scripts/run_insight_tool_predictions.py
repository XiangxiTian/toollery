from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_taichu_toollery_bm25 import (  # noqa: E402
    QMetaBM25Index,
    QMetaDenseIndex,
    TaichuPair,
    dense_embedding_summary,
    generate_or_load_proxy_queries,
    make_dense_embedder,
    retrieval_metrics,
)
from toollery.baselines import ensure_safe_output_path, write_json, write_jsonl  # noqa: E402
from toollery.llm import OpenAICompatibleLLM  # noqa: E402
from toollery.schemas import ToolSpec  # noqa: E402


@dataclass(frozen=True)
class InsightQuery:
    sample_id: str
    query: str


def load_insight_tools(path: str | Path) -> list[ToolSpec]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_tools = data.get("tools") if isinstance(data, dict) else data
    if not isinstance(raw_tools, list):
        raise ValueError(f"{path} must contain a list or an object with a tools list")

    tools: list[ToolSpec] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            raise ValueError(f"tool row {idx} is not an object")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"tool row {idx} is missing name")
        if name in seen:
            raise ValueError(f"duplicate tool name: {name}")
        seen.add(name)
        tools.append(
            ToolSpec(
                name=name,
                description=str(item.get("description", "")).strip(),
                parameters=dict(item.get("parameters") or {}),
                category="insight_tool",
            )
        )
    return tools


def load_unlabeled_queries(path: str | Path) -> list[InsightQuery]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list of query rows")
    queries: list[InsightQuery] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"query row {idx} is not an object")
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        queries.append(InsightQuery(sample_id=f"query_{idx:05d}", query=query))
    return queries


def prediction_rows(
    *,
    tools: list[ToolSpec],
    proxy_pairs: list[TaichuPair],
    queries: list[InsightQuery],
    retriever_name: str,
    top_k: int,
    dense_args: argparse.Namespace | None = None,
) -> list[dict[str, Any]]:
    if retriever_name == "bm25":
        index: Any = QMetaBM25Index(tools, proxy_pairs)
        query_vectors = None
    elif retriever_name == "dense":
        if dense_args is None:
            raise ValueError("dense_args is required for dense retrieval")
        index = QMetaDenseIndex(
            tools,
            proxy_pairs,
            embedder=make_dense_embedder(dense_args),
            embedding_progress_callback=_progress_callback("insight dense docs"),
        )
        query_vectors = index.encode_queries(
            [item.query for item in queries],
            progress_callback=_progress_callback("insight dense queries"),
        )
    else:
        raise ValueError(f"unknown retriever: {retriever_name}")

    rows: list[dict[str, Any]] = []
    for row_idx, item in enumerate(queries, start=1):
        start = time.perf_counter()
        if retriever_name == "dense":
            hits = index.retrieve_from_vector(query_vector=query_vectors[row_idx - 1], top_k=top_k)
        else:
            hits = index.retrieve(item.query, top_k=top_k)
        rows.append(
            {
                "sample_id": item.sample_id,
                "query": item.query,
                "predicted_tools": [hit.tool.name for hit in hits],
                "scores": [hit.score for hit in hits],
                "top_k": min(top_k, len(tools)),
                "retriever": retriever_name,
                "latency_ms": (time.perf_counter() - start) * 1000.0,
            }
        )
        if row_idx == len(queries) or row_idx % 500 == 0:
            print(f"[insight {retriever_name}] predicted {row_idx}/{len(queries)} queries", file=sys.stderr, flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--tools", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--methods", default="bm25,dense")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--proxy-queries")
    parser.add_argument("--generate-proxies", action="store_true")
    parser.add_argument("--generated-queries-out", default="outputs/experiments_v2/taichu_test/insight_tools/generated_queries/deepseek_v4pro.jsonl")
    parser.add_argument("--manual-raw-out", default="outputs/experiments_v2/taichu_test/insight_tools/generated_queries/manual_raw_deepseek_v4pro.jsonl")
    parser.add_argument("--proxy-queries-per-tool", type=int, default=3)
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--force-rebuild-proxies", action="store_true")
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--llm-base-url", default=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--llm-api-key-env", default="LLM_API_KEY")
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-extra-body-json")
    parser.add_argument("--llm-usage-log")
    parser.add_argument("--dense-embedding-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--dense-embedding-model-path")
    parser.add_argument("--dense-embedding-device", default="mps")
    parser.add_argument("--dense-embedding-batch-size", type=int, default=16)
    parser.add_argument("--dense-embedding-max-length", type=int, default=256)
    parser.add_argument("--dense-embedding-pooling", choices=["last", "mean", "cls"], default="last")
    parser.add_argument("--dense-embedding-dtype")
    parser.add_argument("--dense-embedding-trust-remote-code", action="store_true", default=False)
    parser.add_argument("--dense-embedding-local-files-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dense-embedding-implementation",
        choices=["auto", "sentence-transformers", "transformers"],
        default="transformers",
    )
    parser.add_argument("--output-dir", default="outputs/experiments_v2/taichu_test/insight_tools")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_safe_output_path(output_dir)
    tools = load_insight_tools(args.tools)
    queries = load_unlabeled_queries(args.queries)
    if args.limit_queries is not None:
        queries = queries[: args.limit_queries]
    proxy_pairs, proxy_artifact = resolve_proxy_pairs(args, tools)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    summary: dict[str, Any] = {
        "tools": str(Path(args.tools).resolve()),
        "queries": str(Path(args.queries).resolve()),
        "tool_count": len(tools),
        "query_count": len(queries),
        "proxy_artifact": str(proxy_artifact.resolve()) if proxy_artifact else None,
        "proxy_query_count": len(proxy_pairs),
        "proxy_queries_per_tool": args.proxy_queries_per_tool,
        "top_k": args.top_k,
        "has_ground_truth": False,
        "methods": {},
    }

    for method in methods:
        method_dir = output_dir / method
        rows = prediction_rows(
            tools=tools,
            proxy_pairs=proxy_pairs,
            queries=queries,
            retriever_name=method,
            top_k=args.top_k,
            dense_args=args if method == "dense" else None,
        )
        write_jsonl(method_dir / "predictions.jsonl", rows)
        method_summary = {
            "method": method,
            "prediction_count": len(rows),
            "predictions": str((method_dir / "predictions.jsonl").resolve()),
            "avg_latency_ms": _mean(float(row["latency_ms"]) for row in rows),
            "dense_embedding": dense_embedding_summary(args) if method == "dense" else None,
            "metrics_summary": None,
        }
        write_json(method_dir / "summary.json", method_summary)
        summary["methods"][method] = method_summary

    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def resolve_proxy_pairs(args: argparse.Namespace, tools: list[ToolSpec]) -> tuple[list[TaichuPair], Path | None]:
    if args.generate_proxies:
        if args.llm_usage_log:
            os.environ["LLM_USAGE_LOG"] = args.llm_usage_log
        api_key = os.getenv(args.llm_api_key_env)
        if not api_key:
            raise RuntimeError(f"{args.llm_api_key_env} is required for --generate-proxies")
        llm = OpenAICompatibleLLM(
            model=args.llm_model,
            api_key=api_key,
            base_url=args.llm_base_url,
            extra_body=json.loads(args.llm_extra_body_json) if args.llm_extra_body_json else None,
            timeout=args.llm_timeout,
        )
        output_path = Path(args.generated_queries_out)
        pairs = generate_or_load_proxy_queries(
            tools=tools,
            output_path=output_path,
            manual_raw_path=Path(args.manual_raw_out),
            count=args.proxy_queries_per_tool,
            batch_size=args.llm_batch_size,
            llm=llm,
            force_rebuild=args.force_rebuild_proxies,
        )
        return pairs, output_path
    if args.proxy_queries:
        path = Path(args.proxy_queries)
        return _load_proxy_query_pairs(path), path
    raise RuntimeError("Pass --generate-proxies or --proxy-queries")


def _load_proxy_query_pairs(path: Path) -> list[TaichuPair]:
    pairs: list[TaichuPair] = []
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            item = json.loads(line)
            tool_name = str(item.get("tool_name") or item.get("skill_id") or "").strip()
            query = str(item.get("query", "")).strip()
            if tool_name and query:
                pairs.append(TaichuPair(sample_id=f"proxy_{idx:05d}", query=query, expect_intent=tool_name))
    return pairs


def _progress_callback(label: str) -> Any:
    last_done = -1

    def callback(done: int, total: int) -> None:
        nonlocal last_done
        if total <= 0:
            return
        final = done >= total
        step = max(1, total // 20)
        if not final and done - last_done < step:
            return
        last_done = done
        print(f"[{label}] encoded {done}/{total}", file=sys.stderr, flush=True)

    return callback


def _mean(values: Any) -> float | None:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


if __name__ == "__main__":
    main()
